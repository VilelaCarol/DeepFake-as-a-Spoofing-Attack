"""
Baseline de Verificação Facial no YouTube Faces DB usando YOLOv8 + AdaFace (CVLFace).

Pipeline (usando yolo8face_adaface):
  1. Gera pares de vídeos (mesmo/diferente pessoa) automaticamente
  2. Para cada vídeo, amostra N frames e calcula embedding médio (AdaFace)
  3. Similaridade cosseno para verificação
  4. Avaliação: Acurácia, AUC, EER

Estrutura esperada:
  data/YouTubeFaces/aligned_images_DB/
    <Person_Name>/<video_number>/<frame>.jpg
"""

import os
import sys
import time
import glob
import random
import csv
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, accuracy_score

# ==============================================================================
# Adiciona o diretório da nova API ao path
# ==============================================================================
YOLO8FACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolo8face_adaface")
sys.path.insert(0, YOLO8FACE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_lfw import (
    _get_verifier, preprocess_face, get_embedding,
    DEVICE, FACE_SIZE, find_best_threshold,
)

# ==============================================================================
# Configurações
# ==============================================================================
YTF_DATA_DIR = os.path.join("data", "YouTubeFaces", "aligned_images_DB")
RESULTS_DIR = "app/results"

# Número de pares a gerar (metade genuínos, metade impostores)
# Protocolo oficial YTF usa 5000 pares no total (2500 genuínos / 2500 impostores)
NUM_GENUINE_PAIRS = 2500
NUM_IMPOSTOR_PAIRS = 2500

# Quantos frames amostrar de cada vídeo para calcular o embedding médio
FRAMES_TO_SAMPLE = 5

# Thresholds do artigo
THRESHOLDS = [0.45, 0.60]

RANDOM_SEED = 42


# ==============================================================================
# Funções de Apoio
# ==============================================================================
def get_ytf_identities(ytf_dir):
    """
    Retorna dict {nome_pessoa: {video_id: [lista_de_frames]}}
    """
    identities = {}
    if not os.path.isdir(ytf_dir):
        print(f"❌ Diretório não encontrado: {ytf_dir}")
        return identities

    for person_name in sorted(os.listdir(ytf_dir)):
        person_dir = os.path.join(ytf_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        videos = {}
        for vid_id in sorted(os.listdir(person_dir)):
            vid_dir = os.path.join(person_dir, vid_id)
            if not os.path.isdir(vid_dir):
                continue
            frames = sorted(glob.glob(os.path.join(vid_dir, "*.jpg")))
            if len(frames) > 0:
                videos[vid_id] = frames

        if len(videos) > 0:
            identities[person_name] = videos

    return identities


def generate_pairs(identities, num_genuine, num_impostor, seed=RANDOM_SEED):
    """
    Gera pares de vídeos para avaliação.
    Retorna lista de (person1, vid1_id, person2, vid2_id, is_same).

    - Genuínos: mesma pessoa, vídeos diferentes
    - Impostores: pessoas diferentes
    """
    rng = random.Random(seed)

    # Pessoas com 2+ vídeos (para pares genuínos)
    multi_vid_people = {name: vids for name, vids in identities.items() if len(vids) >= 2}
    all_people = list(identities.keys())

    pairs = []

    # Pares genuínos (mesma pessoa, vídeos diferentes)
    multi_names = list(multi_vid_people.keys())
    attempts = 0
    while len([p for p in pairs if p[4] == 1]) < num_genuine and attempts < num_genuine * 20:
        attempts += 1
        name = rng.choice(multi_names)
        vid_ids = list(multi_vid_people[name].keys())
        v1, v2 = rng.sample(vid_ids, 2)
        pairs.append((name, v1, name, v2, 1))

    # Pares impostores (pessoas diferentes)
    attempts = 0
    while len([p for p in pairs if p[4] == 0]) < num_impostor and attempts < num_impostor * 20:
        attempts += 1
        n1, n2 = rng.sample(all_people, 2)
        v1 = rng.choice(list(identities[n1].keys()))
        v2 = rng.choice(list(identities[n2].keys()))
        pairs.append((n1, v1, n2, v2, 0))

    rng.shuffle(pairs)
    return pairs


def get_video_mean_embedding(frames_list, verifier, n_sample=FRAMES_TO_SAMPLE):
    """
    Calcula embedding médio de um vídeo amostrando N frames.
    Usa diretamente as imagens alignadas (já recortadas).
    """
    if len(frames_list) == 0:
        return None

    # Amostragem temporal uniforme
    if len(frames_list) > n_sample:
        idx = np.linspace(0, len(frames_list) - 1, n_sample).astype(int)
        sampled = [frames_list[i] for i in idx]
    else:
        sampled = frames_list

    embeddings = []
    model = verifier.model

    for frame_path in sampled:
        img = cv2.imread(frame_path)
        if img is None:
            continue

        # Imagens do YTF aligned já são rostos recortados — não precisa YOLO
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_tensor = preprocess_face(img_rgb)
        if face_tensor is not None:
            emb = get_embedding(model, face_tensor, DEVICE)
            embeddings.append(emb)

    if len(embeddings) > 0:
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        return mean_emb

    return None


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 70)
    print("BASELINE — Verificação Facial no YouTube Faces DB")
    print("AdaFace CVLFace IR-50 (via HuggingFace) — Vídeo-to-Vídeo")
    print("=" * 70)

    # 1. Verificar dataset
    print(f"\n[1/5] Verificando dataset em: {YTF_DATA_DIR}")
    identities = get_ytf_identities(YTF_DATA_DIR)
    n_people = len(identities)
    n_videos = sum(len(v) for v in identities.values())
    n_multi = sum(1 for v in identities.values() if len(v) >= 2)
    print(f"   Pessoas: {n_people}")
    print(f"   Vídeos totais: {n_videos}")
    print(f"   Pessoas com 2+ vídeos: {n_multi}")

    if n_people == 0:
        print("❌ Dataset vazio! Verifique o caminho.")
        return

    # 2. Gerar pares
    print(f"\n[2/5] Gerando pares de avaliação...")
    pairs = generate_pairs(identities, NUM_GENUINE_PAIRS, NUM_IMPOSTOR_PAIRS)
    n_gen = sum(1 for p in pairs if p[4] == 1)
    n_imp = sum(1 for p in pairs if p[4] == 0)
    print(f"   Total: {len(pairs)} | Genuínos: {n_gen} | Impostores: {n_imp}")

    # 3. Carregar modelo
    print(f"\n[3/5] Carregando AdaFace (device: {DEVICE})...")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    verifier = _get_verifier()

    # 4. Processar pares
    print(f"\n[4/5] Processando pares (vídeo-to-vídeo)...")
    similarities, labels, errors = [], [], 0
    t0 = time.time()

    for i, (p1_name, v1_id, p2_name, v2_id, is_same) in enumerate(pairs):
        frames1 = identities.get(p1_name, {}).get(v1_id, [])
        frames2 = identities.get(p2_name, {}).get(v2_id, [])

        emb1 = get_video_mean_embedding(frames1, verifier)
        emb2 = get_video_mean_embedding(frames2, verifier)

        if emb1 is None or emb2 is None:
            errors += 1
            continue

        sim = float(np.dot(np.squeeze(emb1), np.squeeze(emb2)))
        similarities.append(sim)
        labels.append(is_same)

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(pairs) - i - 1)
            print(f"   {i+1}/{len(pairs)} ({el:.0f}s, ETA: {eta:.0f}s)")

    total_time = time.time() - t0
    print(f"   Concluído em {total_time:.1f}s")
    if errors:
        print(f"   ⚠ {errors} pares ignorados")

    # 5. Avaliação
    print(f"\n[5/5] Avaliação e exportação de resultados")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    sims = np.array(similarities)
    labs = np.array(labels)

    # Melhor threshold
    best_t, best_acc = find_best_threshold(sims, labs)
    fpr, tpr, _ = roc_curve(labs, sims)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

    genuine_scores = sims[labs == 1]
    impostor_scores = sims[labs == 0]

    # Imprimir resumo
    print(f"\n   RESULTADOS NO YTF (YouTube Faces)")
    print(f"   {'─' * 45}")
    print(f"   Pares avaliados:    {len(labs)}")
    print(f"   Melhor threshold:   {best_t:.4f}")
    print(f"   Acurácia:           {best_acc*100:.2f}%")
    print(f"   AUC:                {roc_auc:.4f}")
    print(f"   EER:                {eer*100:.2f}%")
    print(f"   Tempo total:        {total_time:.1f}s")
    print(f"\n   Média Genuínos:     {np.mean(genuine_scores):.4f}")
    print(f"   Média Impostores:   {np.mean(impostor_scores):.4f}")

    # FAR/FRR nos thresholds do artigo
    print(f"\n   FAR / FRR nos thresholds do artigo")
    print(f"   {'Threshold':>12} | {'FAR (%)':>10} | {'FRR (%)':>10}")
    print(f"   {'─'*12}-+-{'─'*10}-+-{'─'*10}")
    threshold_results = []
    for tau in THRESHOLDS:
        far = np.mean(impostor_scores >= tau) * 100
        frr = np.mean(genuine_scores < tau) * 100
        threshold_results.append((tau, far, frr))
        print(f"   {tau:>12.2f} | {far:>10.2f} | {frr:>10.2f}")

    # ── Salvar scores CSV ────────────────────────────────────────
    scores_path = os.path.join(RESULTS_DIR, "ytf_scores.csv")
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "label", "score"])
        for idx, (lab, sim) in enumerate(zip(labs, sims)):
            writer.writerow([idx, int(lab), f"{sim:.6f}"])
    print(f"\n   [A] Scores salvos em: {scores_path}")

    # ── Gráficos ─────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Histograma
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(genuine_scores, bins=60, alpha=0.65, color="#2196F3",
            label=f"Genuínos (n={len(genuine_scores)})", density=True)
    ax.hist(impostor_scores, bins=60, alpha=0.65, color="#F44336",
            label=f"Impostores (n={len(impostor_scores)})", density=True)
    for tau, far, frr in threshold_results:
        ax.axvline(x=tau, linestyle="--", linewidth=1.5,
                   label=f"τ={tau} (FAR={far:.1f}%, FRR={frr:.1f}%)")
    ax.set_xlabel("Similaridade Cosseno", fontsize=12)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.set_title("Distribuição dos Scores — YTF Baseline\nAdaFace CVLFace IR-50 (Vídeo-to-Vídeo)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    hist_path = os.path.join(RESULTS_DIR, "ytf_score_hist_baseline.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"   [B] Histograma salvo em: {hist_path}")

    # Curva ROC
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"AdaFace IR-50 (AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax2.scatter([eer], [1 - eer], color="red", zorder=5,
                label=f"EER = {eer*100:.2f}%")
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.set_title("Curva ROC — YTF\nAdaFace CVLFace IR-50 (Vídeo-to-Vídeo)")
    ax2.legend(loc="lower right"); ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, "ytf_roc_curve.png")
    fig2.savefig(roc_path, dpi=150)
    plt.close(fig2)
    print(f"   [B] Curva ROC salva em: {roc_path}")

    # ── Config experimental ──────────────────────────────────────
    import platform
    config_lines = [
        "=" * 55,
        "CONFIGURAÇÃO EXPERIMENTAL — YTF BASELINE",
        "=" * 55,
        f"Target system    : AdaFace CVLFace IR-50",
        f"Model source     : HuggingFace (minchul/cvlface_adaface_ir50_ms1mv2)",
        f"Dataset/Protocol : YouTube Faces DB (Vídeo-to-Vídeo)",
        f"Frames amostrados: {FRAMES_TO_SAMPLE} por vídeo",
        f"Thresholds usados: {THRESHOLDS}",
        f"Hardware         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"",
        f"Total de pares   : {len(labs)}",
        f"Genuínos         : {int(labs.sum())}",
        f"Impostores       : {len(labs) - int(labs.sum())}",
        f"Pares ignorados  : {errors}",
        f"",
        f"OS               : {platform.system()} {platform.release()}",
        f"Python           : {platform.python_version()}",
        f"PyTorch          : {torch.__version__}",
        f"",
        "=" * 55,
        "RESULTADOS",
        "=" * 55,
        f"Melhor threshold : {best_t:.4f}",
        f"Acurácia         : {best_acc*100:.2f}%",
        f"AUC              : {roc_auc:.4f}",
        f"EER              : {eer*100:.2f}%",
        f"Tempo total (s)  : {total_time:.1f}",
        f"",
        f"Média Genuínos   : {np.mean(genuine_scores):.4f}",
        f"Média Impostores : {np.mean(impostor_scores):.4f}",
        f"",
    ]
    for tau, far, frr in threshold_results:
        config_lines.append(f"τ={tau:.2f}: FAR={far:.2f}%, FRR={frr:.2f}%")

    config_path = os.path.join(RESULTS_DIR, "ytf_experimental_config.txt")
    with open(config_path, "w") as f:
        f.write("\n".join(config_lines) + "\n")
    print(f"   [C] Config experimental salva em: {config_path}")

    print(f"\n{'=' * 70}")
    print("Baseline YTF finalizado! Resultados em: app/results/")
    print("  ytf_scores.csv")
    print("  ytf_score_hist_baseline.png")
    print("  ytf_roc_curve.png")
    print("  ytf_experimental_config.txt")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
