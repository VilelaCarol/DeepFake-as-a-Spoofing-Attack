"""
================================================================================
COMPUTE ASR OFFLINE — Cálculo do ASR pós-execução
================================================================================
Lê os frames salvos pelo OfflineFrameSaver (do realtime_api_server.py),
extrai embeddings via AdaFace, calcula similaridade cosseno contra o
embedding do atacante, e gera tabela completa com:

  - Similaridade por frame
  - Média / Desvio Padrão da similaridade
  - ASR para múltiplos thresholds (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
  - Tabela formatada no terminal + CSV de saída

Uso:
  python compute_asr_offline.py \
    --frames-dir results/realtime_frames \
    --source-image data/dataset_pessoal/originais/victorsorrindoclaro.jpeg

  Ou simplesmente (usa defaults):
  python compute_asr_offline.py
================================================================================
"""

import os
import sys
import csv
import glob
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

# ==============================================================================
# Path da API YOLOv8 + AdaFace
# ==============================================================================
YOLO8FACE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "yolo8face_adaface"
)
sys.path.insert(0, YOLO8FACE_DIR)

from src import AdaFaceVerifier

CONFIG_PATH = os.path.join(YOLO8FACE_DIR, "configs", "config.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Thresholds para avaliar ASR
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def init_verifier() -> AdaFaceVerifier:
    """Carrega o modelo AdaFace."""
    print("🚀 Carregando AdaFaceVerifier (YOLOv8 + AdaFace CVLFace)...")
    old_cwd = os.getcwd()
    os.chdir(YOLO8FACE_DIR)
    try:
        verifier = AdaFaceVerifier(CONFIG_PATH)
    finally:
        os.chdir(old_cwd)
    print(f"✅ Modelo carregado em: {verifier.device}")
    return verifier


def extract_reference_embedding(verifier: AdaFaceVerifier, image_path: str) -> np.ndarray:
    """Extrai o embedding de referência do atacante."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem do atacante não encontrada: {image_path}")
    emb_tensor = verifier.get_embedding(img)
    if emb_tensor is None:
        raise RuntimeError(f"Nenhuma face detectada na imagem do atacante: {image_path}")
    embedding = emb_tensor.cpu().numpy()
    print(f"✅ Embedding de referência extraído — shape: {embedding.shape}")
    return embedding


def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Similaridade cosseno entre dois embeddings."""
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Calcula o Intervalo de Confiança 95% (Wilson Score) para uma proporção."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denominator
    return (center - half_width) * 100, (center + half_width) * 100

def run_offline_asr(frames_dir: str, source_image: str, output_csv: str = None):
    """
    Pipeline principal de cálculo offline do ASR.
    """
    print("\n" + "=" * 70)
    print("📊 COMPUTE ASR OFFLINE — Cálculo pós-execução")
    print("=" * 70)

    # ── 1. Encontra frames salvos ─────────────────────────────────────────
    pattern = os.path.join(frames_dir, "frame_*.jpg")
    frame_files = sorted(glob.glob(pattern))
    if not frame_files:
        print(f"❌ Nenhum frame encontrado em: {frames_dir}")
        print(f"   Pattern: {pattern}")
        print(f"   Execute o realtime_api_server.py primeiro para salvar frames!")
        return

    total_frames = len(frame_files)
    print(f"\n📂 Frames encontrados: {total_frames}")
    print(f"   Diretório: {frames_dir}")

    # ── 2. Carrega modelo e referência ────────────────────────────────────
    verifier = init_verifier()
    ref_emb = extract_reference_embedding(verifier, source_image)

    # ── 3. Processa cada frame ────────────────────────────────────────────
    print(f"\n⏳ Processando {total_frames} frames...\n")
    results = []
    failed = 0
    t0 = time.time()

    for i, fpath in enumerate(frame_files):
        fname = os.path.basename(fpath)
        # Extrai ID e modo do nome do arquivo: frame_000005_ATAQUE.jpg
        parts = fname.replace(".jpg", "").split("_")
        frame_id = int(parts[1]) if len(parts) >= 2 else i
        mode = parts[2] if len(parts) >= 3 else "UNKNOWN"

        img = cv2.imread(fpath)
        if img is None:
            failed += 1
            continue

        emb_tensor = verifier.get_embedding(img)
        if emb_tensor is None:
            # Nenhuma face detectada neste frame
            results.append({
                "frame_id": frame_id,
                "mode": mode,
                "file": fname,
                "similarity": None,
                "face_detected": False,
            })
            continue

        emb = emb_tensor.cpu().numpy()
        sim = compute_similarity(emb, ref_emb)
        results.append({
            "frame_id": frame_id,
            "mode": mode,
            "file": fname,
            "similarity": sim,
            "face_detected": True,
        })

        # Progresso
        if (i + 1) % 20 == 0 or (i + 1) == total_frames:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total_frames - i - 1) / rate if rate > 0 else 0
            print(f"   [{i+1:4d}/{total_frames}] sim={sim:.4f} | "
                  f"{rate:.1f} frames/s | ETA: {eta:.0f}s")

    elapsed_total = time.time() - t0

    # ── 4. Filtra resultados válidos (face detectada) ─────────────────────
    valid = [r for r in results if r["face_detected"] and r["similarity"] is not None]
    no_face = [r for r in results if not r["face_detected"]]
    sims = np.array([r["similarity"] for r in valid])

    if len(sims) == 0:
        print("❌ Nenhum embedding válido extraído dos frames!")
        return

    # ── 5. Estatísticas ───────────────────────────────────────────────────
    sim_mean = float(np.mean(sims))
    sim_std = float(np.std(sims))
    sim_median = float(np.median(sims))
    sim_min = float(np.min(sims))
    sim_max = float(np.max(sims))
    sim_q25 = float(np.percentile(sims, 25))
    sim_q75 = float(np.percentile(sims, 75))

    # ASR por threshold
    asr_table = {}
    for tau in THRESHOLDS:
        accepted = int(np.sum(sims >= tau))
        asr = 100.0 * accepted / len(sims)
        lower, upper = wilson_ci(accepted, len(sims))
        asr_table[tau] = {
            "accepted": accepted, 
            "total": len(sims), 
            "asr": asr,
            "ci_lower": lower,
            "ci_upper": upper
        }

    # ── 6. Exibe tabela no terminal ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 RESULTADOS — ATTACK SUCCESS RATE (ASR) OFFLINE")
    print("=" * 80)

    print(f"\n{'─' * 50}")
    print(f"  📐 ESTATÍSTICAS DE SIMILARIDADE COSSENO")
    print(f"{'─' * 50}")
    print(f"  Frames processados : {total_frames}")
    print(f"  Faces detectadas   : {len(valid)} ({100*len(valid)/total_frames:.1f}%)")
    print(f"  Sem face           : {len(no_face)}")
    print(f"  Falhas de leitura  : {failed}")
    print(f"{'─' * 50}")
    print(f"  Média (μ)          : {sim_mean:.6f}")
    print(f"  Desvio Padrão (σ)  : {sim_std:.6f}")
    print(f"  Mediana            : {sim_median:.6f}")
    print(f"  Mínimo             : {sim_min:.6f}")
    print(f"  Máximo             : {sim_max:.6f}")
    print(f"  Q1 (25%)           : {sim_q25:.6f}")
    print(f"  Q3 (75%)           : {sim_q75:.6f}")
    print(f"  IQR                : {sim_q75 - sim_q25:.6f}")
    print(f"{'─' * 50}")

    print(f"\n{'─' * 80}")
    print(f"  🎯 ASR POR THRESHOLD")
    print(f"{'─' * 80}")
    print(f"  {'Threshold (τ)':>14s} | {'Aceitos':>8s} | {'Total':>6s} | {'ASR (%)':>8s} | {'IC 95% (Wilson)':>18s}")
    print(f"  {'─' * 14}─┼─{'─' * 8}─┼─{'─' * 6}─┼─{'─' * 8}─┼─{'─' * 20}")
    for tau in THRESHOLDS:
        info = asr_table[tau]
        marker = " ◀" if tau == 0.45 else ""
        ci_str = f"[{info['ci_lower']:>5.2f}%, {info['ci_upper']:>5.2f}%]"
        print(f"  {tau:>14.2f} | {info['accepted']:>8d} | {info['total']:>6d} | "
              f"{info['asr']:>7.2f}% | {ci_str:<18s}{marker}")
    print(f"{'─' * 80}")

    print(f"\n  ⏱  Tempo total de processamento: {elapsed_total:.1f}s")
    print(f"  📈 Velocidade: {total_frames / elapsed_total:.1f} frames/s")

    # ── 7. Salva CSV detalhado ────────────────────────────────────────────
    if output_csv is None:
        output_csv = os.path.join(RESULTS_DIR, "realtime_asr_offline.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "mode", "file", "face_detected", "similarity",
                         "match_030", "match_035", "match_040", "match_045",
                         "match_050", "match_055", "match_060"])
        for r in results:
            sim = r["similarity"]
            if sim is not None:
                matches = [int(sim >= tau) for tau in THRESHOLDS]
            else:
                matches = [0] * len(THRESHOLDS)
            writer.writerow([
                r["frame_id"], r["mode"], r["file"],
                int(r["face_detected"]),
                f"{sim:.6f}" if sim is not None else "",
                *matches
            ])

    print(f"\n  📄 CSV detalhado salvo em: {output_csv}")

    # ── 8. Salva resumo estatístico ───────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "realtime_asr_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"REALTIME ASR OFFLINE ANALYSIS\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Frames dir: {frames_dir}\n")
        f.write(f"Source image: {source_image}\n\n")

        f.write(f"SIMILARITY STATISTICS\n")
        f.write(f"{'-' * 50}\n")
        f.write(f"Total frames processed: {total_frames}\n")
        f.write(f"Valid faces detected: {len(valid)}\n")
        f.write(f"Mean (μ): {sim_mean:.6f}\n")
        f.write(f"Std Dev (σ): {sim_std:.6f}\n")
        f.write(f"Median: {sim_median:.6f}\n")
        f.write(f"Min: {sim_min:.6f}\n")
        f.write(f"Max: {sim_max:.6f}\n")
        f.write(f"Q1 (25%): {sim_q25:.6f}\n")
        f.write(f"Q3 (75%): {sim_q75:.6f}\n\n")

        f.write(f"ASR BY THRESHOLD\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"{'Threshold (τ)':>14s} | {'Aceitos':>8s} | {'Total':>6s} | {'ASR (%)':>8s} | {'IC 95% (Wilson)':>18s}\n")
        for tau in THRESHOLDS:
            info = asr_table[tau]
            ci_str = f"[{info['ci_lower']:>5.2f}%, {info['ci_upper']:>5.2f}%]"
            f.write(f"{tau:>14.2f} | {info['accepted']:>8d} | {info['total']:>6d} | "
                    f"{info['asr']:>7.2f}% | {ci_str:<18s}\n")

    print(f"  📄 Resumo salvo em: {summary_path}")
    print("\n" + "=" * 70)

    return {
        "mean": sim_mean,
        "std": sim_std,
        "median": sim_median,
        "min": sim_min,
        "max": sim_max,
        "asr_table": asr_table,
        "total_frames": total_frames,
        "valid_faces": len(valid),
    }


# ==============================================================================
# Entry point
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute ASR Offline — Cálculo pós-execução do Attack Success Rate"
    )
    p.add_argument(
        "--frames-dir", type=str,
        default=os.path.join(RESULTS_DIR, "realtime_frames"),
        help="Diretório com os frames salvos pelo OfflineFrameSaver"
    )
    p.add_argument(
        "--source-image", type=str,
        default="data/dataset_pessoal/originais/victorsorrindoclaro.jpeg",
        help="Imagem do atacante (referência para o embedding)"
    )
    p.add_argument(
        "--output-csv", type=str, default=None,
        help="Caminho para o CSV de saída (default: results/realtime_asr_offline.csv)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_offline_asr(
        frames_dir=args.frames_dir,
        source_image=args.source_image,
        output_csv=args.output_csv,
    )
