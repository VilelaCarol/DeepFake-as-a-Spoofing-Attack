"""
Experimento de Ataque com Deepfake no YouTube Faces DB.

Mesmo raciocínio do run_swaps_lfw.py, agora aplicado ao dataset YTF:

  1. Seleciona pares de identidades diferentes do YTF
  2. Para cada par, pega frames do vídeo-alvo e aplica face swap com FaceFusion
  3. Compara o swap (vídeo falso) com o vídeo original da pessoa-source usando AdaFace
  4. Se a similaridade >= threshold → o deepfake enganou a biometria

Gera gráficos e relatórios de ASR (Attack Success Rate) para comparação com o baseline.
"""

import os
import sys
import math
import random
import time
import subprocess
import glob
import csv
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch

# Caminhos e imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_lfw import (
    _get_verifier, preprocess_face, get_embedding,
    DEVICE, FACE_SIZE,
)
from baseline_ytf import (
    YTF_DATA_DIR, RESULTS_DIR,
    get_ytf_identities, get_video_mean_embedding,
    FRAMES_TO_SAMPLE,
)

# ==============================================================================
# Configurações do Experimento
# ==============================================================================
SWAPS_DIR = os.path.join("data", "YouTubeFaces", "swaps")

# Número de ataques de vídeo a gerar
NUM_ATTACKS = 1000

# Thresholds para avaliar
THRESHOLDS = [0.45, 0.60]

RANDOM_SEED = 42


# ==============================================================================
# Funções
# ==============================================================================
def generate_attack_pairs(identities, num_attacks, seed=RANDOM_SEED):
    """
    Gera pares de ataque: (source_person, source_vid, target_person, target_vid)
    Source = atacante (cujo rosto será colado)
    Target = vítima (cujo corpo/vídeo será usado)
    
    Sempre pessoas DIFERENTES.
    """
    rng = random.Random(seed)
    all_people = list(identities.keys())
    pairs = []
    attempts = 0

    while len(pairs) < num_attacks and attempts < num_attacks * 20:
        attempts += 1
        src_name = rng.choice(all_people)
        tgt_name = rng.choice(all_people)

        if src_name == tgt_name:
            continue

        src_vid = rng.choice(list(identities[src_name].keys()))
        tgt_vid = rng.choice(list(identities[tgt_name].keys()))

        pairs.append((src_name, src_vid, tgt_name, tgt_vid))

    return pairs


def run_facefusion_swap(src_path, tgt_path, out_path, cwd="facefusion"):
    """Roda o FaceFusion para gerar um swap. Retorna True se sucesso."""
    cmd = [
        "python", "facefusion.py", "headless-run",
        "--processors", "face_swapper",
        "--face-swapper-model", "inswapper_128",
        "-s", os.path.abspath(src_path),
        "-t", os.path.abspath(tgt_path),
        "-o", os.path.abspath(out_path),
        "--execution-providers", "cuda"
    ]
    try:
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=120)
        return True
    except subprocess.CalledProcessError as e:
        print(f"      ⚠ Erro no FaceFusion: {e.stderr[:200] if e.stderr else 'sem detalhes'}")
        return False
    except subprocess.TimeoutExpired:
        print(f"      ⚠ Timeout no FaceFusion (>120s)")
        return False


def get_swap_mean_embedding(swap_frame_paths, verifier):
    """Calcula embedding médio a partir dos frames swapped."""
    embeddings = []
    model = verifier.model

    for fp in swap_frame_paths:
        if not os.path.exists(fp):
            continue
        img = cv2.imread(fp)
        if img is None:
            continue

        # Imagens do swap mantêm o mesmo tamanho das do YTF (já alinhadas)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_tensor = preprocess_face(img_rgb)
        if face_tensor is not None:
            emb = get_embedding(model, face_tensor, DEVICE)
            embeddings.append(emb)

    if len(embeddings) > 0:
        mean_emb = np.mean(embeddings, axis=0)
        return mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    return None


def wilson_ci(n_success, n_total, z=1.96):
    """Intervalo de confiança de Wilson (95%) para proporção."""
    if n_total == 0:
        return 0, 0
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    return max(0, center - margin), min(1, center + margin)


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 90)
    print("🚀 EXPERIMENTO DE ATAQUE DEEPFAKE NO YOUTUBE FACES DB".center(90))
    print("Face Swap (Vídeo) + Avaliação Biométrica (AdaFace)".center(90))
    print("=" * 90)

    # ──────────────────────────────────────────────────────────────
    # PASSO 0: Verificar dataset e preparar
    # ──────────────────────────────────────────────────────────────
    print("\n📦 Passo 0/4: Verificando dataset...")
    identities = get_ytf_identities(YTF_DATA_DIR)
    print(f"   Pessoas: {len(identities)} | Vídeos: {sum(len(v) for v in identities.values())}")

    os.makedirs(SWAPS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # PASSO 1: Gerar pares de ataque
    # ──────────────────────────────────────────────────────────────
    print(f"\n🔍 Passo 1/4: Gerando {NUM_ATTACKS} pares de ataque...")
    attack_pairs = generate_attack_pairs(identities, NUM_ATTACKS)
    print(f"   Gerados {len(attack_pairs)} pares de ataque")
    print(f"   Seed: {RANDOM_SEED}")

    print(f"\n   Primeiros 5 pares:")
    for i, (sn, sv, tn, tv) in enumerate(attack_pairs[:5]):
        print(f"     {i+1}. [Rosto] {sn}/{sv} → [Corpo] {tn}/{tv}")

    # ──────────────────────────────────────────────────────────────
    # PASSO 2: Gerar face swaps com FaceFusion
    # ──────────────────────────────────────────────────────────────
    print(f"\n🎭 Passo 2/4: Gerando face swaps com FaceFusion...")
    print("   Para cada par, fazemos swap em N frames do vídeo-alvo\n")

    verifier = _get_verifier()
    t0 = time.time()

    attack_results = []  # (src_name, src_vid, tgt_name, tgt_vid, swap_frame_paths, src_frames)

    for pair_idx, (src_name, src_vid, tgt_name, tgt_vid) in enumerate(attack_pairs):
        print(f"   [{pair_idx+1:3d}/{len(attack_pairs)}] {src_name}/{src_vid} → {tgt_name}/{tgt_vid}")

        src_frames = identities.get(src_name, {}).get(src_vid, [])
        tgt_frames = identities.get(tgt_name, {}).get(tgt_vid, [])

        if not src_frames or not tgt_frames:
            print(f"      ⚠ Frames não encontrados, pulando...")
            continue

        # Pega o frame do meio do atacante como referência
        src_ref = src_frames[len(src_frames) // 2]

        # Amostra N frames do alvo para aplicar swap
        if len(tgt_frames) > FRAMES_TO_SAMPLE:
            idx = np.linspace(0, len(tgt_frames) - 1, FRAMES_TO_SAMPLE).astype(int)
            sampled_tgt = [tgt_frames[i] for i in idx]
        else:
            sampled_tgt = tgt_frames

        # Gerar swap em cada frame amostrado
        swap_paths = []
        for j, tgt_frame in enumerate(sampled_tgt):
            safe_src = f"{src_name}_{src_vid}"
            safe_tgt = f"{tgt_name}_{tgt_vid}"
            out_name = f"swap_{safe_src}_on_{safe_tgt}_{j}.jpg"
            out_path = os.path.join(SWAPS_DIR, out_name)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                swap_paths.append(out_path)
                continue

            success = run_facefusion_swap(src_ref, tgt_frame, out_path)
            if success:
                swap_paths.append(out_path)

        n_ok = len(swap_paths)
        if n_ok > 0:
            print(f"      ✅ {n_ok}/{len(sampled_tgt)} frames com swap")
        else:
            print(f"      ❌ Nenhum swap gerado")

        attack_results.append((src_name, src_vid, tgt_name, tgt_vid, swap_paths, src_frames))

    elapsed_swap = time.time() - t0
    n_with_swaps = sum(1 for r in attack_results if len(r[4]) > 0)
    print(f"\n   Swaps concluídos em {elapsed_swap:.1f}s")
    print(f"   ✅ Ataques com swap: {n_with_swaps}/{len(attack_results)}")

    # ──────────────────────────────────────────────────────────────
    # PASSO 3: Avaliar os swaps com AdaFace
    # ──────────────────────────────────────────────────────────────
    print(f"\n🧠 Passo 3/4: Avaliando swaps com AdaFace...")
    print("   Comparando cada vídeo deepfake com o vídeo original do atacante...\n")

    evaluation_results = []

    for i, (src_name, src_vid, tgt_name, tgt_vid, swap_paths, src_frames) in enumerate(attack_results):
        if len(swap_paths) == 0:
            continue

        # Embedding do atacante (vídeo original)
        emb_src = get_video_mean_embedding(src_frames, verifier)
        # Embedding do deepfake (frames swapped)
        emb_swap = get_swap_mean_embedding(swap_paths, verifier)

        if emb_src is None or emb_swap is None:
            continue

        sim = float(np.dot(np.squeeze(emb_src), np.squeeze(emb_swap)))

        evaluation_results.append({
            "source_name": src_name,
            "source_vid": src_vid,
            "target_name": tgt_name,
            "target_vid": tgt_vid,
            "similarity": sim,
            "n_swap_frames": len(swap_paths),
        })

        status_045 = "🚨 ENGANOU" if sim >= 0.45 else "🛡️ BARRADO"
        status_060 = "🚨 ENGANOU" if sim >= 0.60 else "🛡️ BARRADO"
        print(f"   [{i+1:3d}] {src_name:25s} → {tgt_name:25s}  "
              f"sim={sim:.4f}  τ0.45:{status_045}  τ0.60:{status_060}")

    # ──────────────────────────────────────────────────────────────
    # PASSO 4: Relatório Final
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(" 📊 RESULTADOS DO ATAQUE DEEPFAKE NO YOUTUBE FACES DB ".center(100, "═"))
    print(f"{'='*100}")

    if not evaluation_results:
        print("\n❌ Nenhum swap foi avaliado. Verifique se o FaceFusion está funcionando.")
        return

    sims = np.array([r["similarity"] for r in evaluation_results])
    n_total = len(sims)

    # ── Taxa de sucesso do ataque ────────────────────────────────
    print(f"\n{'='*80}")
    print(" 📈 TAXA DE SUCESSO DO ATAQUE (Attack Success Rate) ".center(80, "═"))
    print(f"{'='*80}")

    for tau in THRESHOLDS:
        n_fooled = int(np.sum(sims >= tau))
        asr = n_fooled / n_total * 100
        ci_low, ci_high = wilson_ci(n_fooled, n_total)
        print(f"\n   Threshold τ = {tau:.2f}:")
        print(f"     Swaps que ENGANARAM a IA: {n_fooled}/{n_total} ({asr:.1f}%)")
        print(f"     Swaps BARRADOS pela IA:   {n_total - n_fooled}/{n_total} ({100 - asr:.1f}%)")
        print(f"     IC 95% (Wilson):          [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")

    # ── Estatísticas descritivas ─────────────────────────────────
    print(f"\n   Estatísticas de Similaridade dos Swaps:")
    print(f"     Média:   {np.mean(sims):.4f}")
    print(f"     Mediana: {np.median(sims):.4f}")
    print(f"     Desvio:  {np.std(sims):.4f}")
    print(f"     Mín:     {np.min(sims):.4f}")
    print(f"     Máx:     {np.max(sims):.4f}")
    print(f"     Q1 (25%): {np.percentile(sims, 25):.4f}")
    print(f"     Q3 (75%): {np.percentile(sims, 75):.4f}")

    # ── Testes estatísticos (se baseline existir) ────────────────
    try:
        from scipy import stats as sp_stats
        baseline_csv = os.path.join(RESULTS_DIR, "ytf_scores.csv")
        if os.path.exists(baseline_csv):
            bl_df = pd.read_csv(baseline_csv)
            genuine_sc = bl_df[bl_df["label"] == 1]["score"].values
            impostor_sc = bl_df[bl_df["label"] == 0]["score"].values

            print(f"\n{'='*80}")
            print(" 🔬 TESTES ESTATÍSTICOS ".center(80, "═"))
            print(f"{'='*80}")

            ks_stat_g, ks_p_g = sp_stats.ks_2samp(sims, genuine_sc)
            print(f"\n   Kolmogorov-Smirnov (Swaps vs Genuínos):")
            print(f"     KS statistic = {ks_stat_g:.4f}")
            print(f"     p-value      = {ks_p_g:.2e}")

            ks_stat_i, ks_p_i = sp_stats.ks_2samp(sims, impostor_sc)
            print(f"\n   Kolmogorov-Smirnov (Swaps vs Impostores):")
            print(f"     KS statistic = {ks_stat_i:.4f}")
            print(f"     p-value      = {ks_p_i:.2e}")

            u_stat, u_p = sp_stats.mannwhitneyu(sims, genuine_sc, alternative='two-sided')
            print(f"\n   Mann-Whitney U (Swaps vs Genuínos):")
            print(f"     U statistic  = {u_stat:.0f}")
            print(f"     p-value      = {u_p:.2e}")

            pooled_std = np.sqrt((np.std(sims)**2 + np.std(genuine_sc)**2) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(sims) - np.mean(genuine_sc)) / pooled_std
                print(f"\n   Cohen's d (Swaps vs Genuínos): {cohens_d:.4f}")
                if abs(cohens_d) < 0.2:    print(f"     → Efeito NEGLIGÍVEL")
                elif abs(cohens_d) < 0.5:  print(f"     → Efeito PEQUENO")
                elif abs(cohens_d) < 0.8:  print(f"     → Efeito MÉDIO")
                else:                       print(f"     → Efeito GRANDE")

            print(f"\n   Resumo: Média Genuínos={np.mean(genuine_sc):.4f}  "
                  f"Média Impostores={np.mean(impostor_sc):.4f}  "
                  f"Média Swaps={np.mean(sims):.4f}")
        else:
            print(f"\n   ⚠ ytf_scores.csv não encontrado — rode baseline_ytf.py primeiro para comparação estatística.")
    except ImportError:
        print(f"\n   ⚠ scipy não instalado — testes estatísticos pulados.")

    # ── Salvar CSV ───────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "ytf_swap_attack_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "attack_id", "source_name", "source_vid",
            "target_name", "target_vid",
            "similarity", "n_swap_frames",
            "fooled_045", "fooled_060"
        ])
        for idx, r in enumerate(evaluation_results):
            writer.writerow([
                idx, r["source_name"], r["source_vid"],
                r["target_name"], r["target_vid"],
                f"{r['similarity']:.6f}", r["n_swap_frames"],
                int(r["similarity"] >= 0.45),
                int(r["similarity"] >= 0.60),
            ])
    print(f"\n   📁 Resultados salvos em: {csv_path}")

    # ── Gráficos ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Histograma dos scores de ataque
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(sims, bins=40, alpha=0.7, color="#E91E63", edgecolor="white",
                label=f"Swaps (n={n_total})")
        for tau in THRESHOLDS:
            n_fooled = int(np.sum(sims >= tau))
            asr = n_fooled / n_total * 100
            ax.axvline(x=tau, linestyle="--", linewidth=2,
                       label=f"τ={tau} (ASR={asr:.1f}%)")
        ax.set_xlabel("Similaridade Cosseno (Swap vs Original)", fontsize=12)
        ax.set_ylabel("Quantidade", fontsize=12)
        ax.set_title(f"Distribuição dos Scores — Ataque Deepfake no YTF (n={n_total})\n"
                      "FaceFusion (inswapper_128) + AdaFace IR-50 (Vídeo)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        hist_path = os.path.join(RESULTS_DIR, "ytf_swap_attack_histogram.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"   📊 Histograma salvo em: {hist_path}")

        # 2. Comparação Baseline vs Ataque
        baseline_csv = os.path.join(RESULTS_DIR, "ytf_scores.csv")
        if os.path.exists(baseline_csv):
            baseline_df = pd.read_csv(baseline_csv)
            genuine_scores = baseline_df[baseline_df["label"] == 1]["score"].values
            impostor_scores = baseline_df[baseline_df["label"] == 0]["score"].values

            fig2, ax2 = plt.subplots(figsize=(11, 6))
            ax2.hist(genuine_scores, bins=60, alpha=0.55, color="#2196F3", density=True,
                     label=f"Genuínos YTF (n={len(genuine_scores)})")
            ax2.hist(impostor_scores, bins=60, alpha=0.55, color="#4CAF50", density=True,
                     label=f"Impostores YTF (n={len(impostor_scores)})")
            ax2.hist(sims, bins=40, alpha=0.65, color="#F44336", density=True,
                     label=f"Deepfakes/Swaps (n={n_total})")
            for tau in THRESHOLDS:
                ax2.axvline(x=tau, linestyle="--", linewidth=1.5, color="black",
                            label=f"τ={tau}")
            ax2.set_xlabel("Similaridade Cosseno", fontsize=12)
            ax2.set_ylabel("Densidade", fontsize=12)
            ax2.set_title(f"Baseline vs Ataque Deepfake — YTF (n={n_total} swaps)\n"
                          "Onde os swaps caem na distribuição?", fontsize=13)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            comp_path = os.path.join(RESULTS_DIR, "ytf_baseline_vs_attack.png")
            fig2.savefig(comp_path, dpi=150)
            plt.close(fig2)
            print(f"   📊 Comparação baseline vs ataque: {comp_path}")

            # 3. Box plot comparativo
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            box_data = [impostor_scores, sims, genuine_scores]
            box_labels = [f"Impostores\n(n={len(impostor_scores)})",
                          f"Deepfakes\n(n={n_total})",
                          f"Genuínos\n(n={len(genuine_scores)})"]
            bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True,
                             medianprops=dict(color='black', linewidth=2))
            colors = ['#4CAF50', '#F44336', '#2196F3']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for tau in THRESHOLDS:
                ax4.axhline(y=tau, linestyle='--', color='gray', alpha=0.7,
                            label=f'τ={tau}')
            ax4.set_ylabel("Similaridade Cosseno", fontsize=12)
            ax4.set_title("Distribuição de Scores: Impostores vs Deepfakes vs Genuínos — YTF", fontsize=13)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
            fig4.tight_layout()
            box_path = os.path.join(RESULTS_DIR, "ytf_boxplot_comparison.png")
            fig4.savefig(box_path, dpi=150)
            plt.close(fig4)
            print(f"   📊 Box plot salvo em: {box_path}")

        # 4. Curva ASR vs Threshold
        tau_range = np.arange(0.05, 0.95, 0.01)
        asr_values = [np.mean(sims >= t) * 100 for t in tau_range]
        ci_low_v = [wilson_ci(int(np.sum(sims >= t)), n_total)[0] * 100 for t in tau_range]
        ci_high_v = [wilson_ci(int(np.sum(sims >= t)), n_total)[1] * 100 for t in tau_range]

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(tau_range, asr_values, color="#D32F2F", lw=2.5, label="ASR (%)")
        ax3.fill_between(tau_range, ci_low_v, ci_high_v,
                         alpha=0.2, color="#D32F2F", label="IC 95% (Wilson)")
        for tau in THRESHOLDS:
            asr_at = np.mean(sims >= tau) * 100
            ax3.axvline(x=tau, linestyle="--", color="black", alpha=0.6)
            ax3.scatter([tau], [asr_at], color="black", zorder=5, s=60)
            ax3.annotate(f"τ={tau}\nASR={asr_at:.1f}%",
                         xy=(tau, asr_at), xytext=(tau + 0.04, asr_at + 3),
                         fontsize=9, fontweight='bold',
                         arrowprops=dict(arrowstyle="->", color="black"))
        ax3.set_xlabel("Threshold (τ)", fontsize=12)
        ax3.set_ylabel("Attack Success Rate — ASR (%)", fontsize=12)
        ax3.set_title(f"ASR vs Threshold — Ataque Deepfake no YTF (n={n_total})\n"
                       "FaceFusion (inswapper_128) + AdaFace IR-50 (Vídeo)", fontsize=13)
        ax3.set_ylim(-2, 105)
        ax3.legend(fontsize=10, loc="lower left")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        asr_path = os.path.join(RESULTS_DIR, "ytf_asr_vs_threshold.png")
        fig3.savefig(asr_path, dpi=150)
        plt.close(fig3)
        print(f"   📊 Curva ASR salva em: {asr_path}")

    except Exception as e:
        print(f"   ⚠ Erro gerando gráficos: {e}")
        import traceback
        traceback.print_exc()

    # ── Conclusão ────────────────────────────────────────────────
    asr_045 = np.mean(sims >= 0.45) * 100
    asr_060 = np.mean(sims >= 0.60) * 100
    n_fooled_045 = int(np.sum(sims >= 0.45))
    n_fooled_060 = int(np.sum(sims >= 0.60))
    ci_lo_045, ci_hi_045 = wilson_ci(n_fooled_045, n_total)
    ci_lo_060, ci_hi_060 = wilson_ci(n_fooled_060, n_total)

    conclusion = f"""
{'=' * 70}
CONCLUSÃO DO ATAQUE DEEPFAKE — YOUTUBE FACES DB
{'=' * 70}

Foram gerados {n_total} ataques de face swap em vídeos utilizando
FaceFusion (inswapper_128) sobre o dataset YouTube Faces DB.

Para cada ataque:
  - Selecionamos o frame do meio do vídeo do atacante (source)
  - Aplicamos face swap em {FRAMES_TO_SAMPLE} frames do vídeo-alvo (target)
  - Calculamos o embedding médio do vídeo deepfake resultante
  - Comparamos com o embedding médio do vídeo original do atacante

  TAXA DE SUCESSO DO ATAQUE (Attack Success Rate):
  
    τ = 0.45: ASR = {asr_045:.1f}% ({n_fooled_045}/{n_total})  IC 95%: [{ci_lo_045*100:.1f}%, {ci_hi_045*100:.1f}%]
    τ = 0.60: ASR = {asr_060:.1f}% ({n_fooled_060}/{n_total})  IC 95%: [{ci_lo_060*100:.1f}%, {ci_hi_060*100:.1f}%]

  Estatísticas de Similaridade:
    Média:   {np.mean(sims):.4f}
    Mediana: {np.median(sims):.4f}
    Desvio:  {np.std(sims):.4f}

INTERPRETAÇÃO:
  - Este experimento testa a robustez do sistema biométrico contra
    ataques deepfake em VÍDEO (vs imagens estáticas no LFW)
  - A comparação com o baseline do YTF permite avaliar se os swaps
    conseguem "enganar" o sistema de verificação facial

{'=' * 70}
"""
    conclusion_path = os.path.join(RESULTS_DIR, "ytf_swap_attack_conclusion.txt")
    with open(conclusion_path, "w") as f:
        f.write(conclusion)
    print(f"   📝 Conclusão salva em: {conclusion_path}")
    print(conclusion)

    print(f"\n{'='*70}")
    print("Experimento finalizado! Arquivos gerados em app/results/:")
    print("  📄 ytf_swap_attack_results.csv      - Detalhes de cada ataque")
    print("  📊 ytf_swap_attack_histogram.png     - Distribuição dos scores")
    print("  📊 ytf_baseline_vs_attack.png        - Baseline vs Ataque")
    print("  📊 ytf_boxplot_comparison.png        - Box plot comparativo")
    print("  📊 ytf_asr_vs_threshold.png          - Curva ASR vs Threshold")
    print("  📝 ytf_swap_attack_conclusion.txt    - Conclusão do experimento")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
