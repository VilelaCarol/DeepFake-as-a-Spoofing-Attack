"""
================================================================================
generate_paper_plots.py
Gera todos os gráficos acadêmicos faltantes para o paper:

  1. Curva ROC combinada (LFW + YTF baseline + ataques)
  2. FAR / FRR vs Threshold (LFW e YTF juntos)
  3. Curva DET (Detection Error Tradeoff)
  4. Curva ASR combinada (LFW vs YTF)
  5. Histograma combinado de scores (4 distribuições)
  6. Matriz de Confusão nos thresholds operacionais
  7. Tabela resumo de métricas (EER, AUC, ASR)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import stats as sp_stats
from scipy.special import erfinv
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES      = os.path.join(BASE, "results")
OUT      = os.path.join(RES, "paper_plots")
os.makedirs(OUT, exist_ok=True)

THRESHOLDS = [0.45, 0.60]

# Paleta consistente
C_GEN   = "#2196F3"   # azul   — genuínos
C_IMP   = "#4CAF50"   # verde  — impostores
C_LFW   = "#E91E63"   # rosa   — ataque LFW
C_YTF   = "#FF9800"   # laranja — ataque YTF
C_BASE  = "#9C27B0"   # roxo   — baseline

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

# ── Carrega dados ─────────────────────────────────────────────────────────────
def load_data():
    lfw_base  = pd.read_csv(os.path.join(RES, "lfw_scores.csv"))
    lfw_atk   = pd.read_csv(os.path.join(RES, "lfw_swap_attack_results.csv"))
    ytf_base  = pd.read_csv(os.path.join(RES, "ytf_scores.csv"))
    ytf_atk   = pd.read_csv(os.path.join(RES, "ytf_swap_attack_results.csv"))

    # Scores por categoria
    lfw_gen  = lfw_base[lfw_base["label"] == 1]["score"].values
    lfw_imp  = lfw_base[lfw_base["label"] == 0]["score"].values
    lfw_sw   = lfw_atk["similarity"].values

    ytf_gen  = ytf_base[ytf_base["label"] == 1]["score"].values
    ytf_imp  = ytf_base[ytf_base["label"] == 0]["score"].values
    ytf_sw   = ytf_atk["similarity"].values

    return lfw_gen, lfw_imp, lfw_sw, ytf_gen, ytf_imp, ytf_sw


def eer_from_roc(fpr, tpr, thresholds):
    """Calcula EER (Equal Error Rate) a partir da curva ROC."""
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    thr = thresholds[idx]
    return eer, thr


def far_frr_curve(genuine, impostor, tau_range):
    """Retorna FAR e FRR para cada threshold."""
    far, frr = [], []
    for tau in tau_range:
        far.append(np.mean(impostor >= tau))   # falso aceite
        frr.append(np.mean(genuine  <  tau))   # falsa rejeição
    return np.array(far), np.array(frr)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CURVA ROC COMBINADA
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_combined(lfw_gen, lfw_imp, lfw_sw, ytf_gen, ytf_imp, ytf_sw):
    fig, ax = plt.subplots(figsize=(8, 7))

    configs = [
        # (scores_neg, scores_pos, label, color, linestyle)
        (lfw_imp,  lfw_gen, "LFW Baseline (Genuíno vs Impostor)",  C_GEN,  "-"),
        (ytf_imp,  ytf_gen, "YTF Baseline (Genuíno vs Impostor)",  C_BASE, "-"),
        (lfw_imp,  lfw_sw,  "LFW Ataque Deepfake (Swap vs Impostor)", C_LFW,  "--"),
        (ytf_imp,  ytf_sw,  "YTF Ataque Deepfake (Swap vs Impostor)", C_YTF,  "--"),
    ]

    for neg, pos, label, color, ls in configs:
        y_true  = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        y_score = np.concatenate([pos, neg])
        fpr, tpr, thr = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        eer, eer_thr = eer_from_roc(fpr, tpr, thr)
        ax.plot(fpr, tpr, color=color, lw=2, ls=ls,
                label=f"{label}\nAUC={roc_auc:.4f} | EER={eer*100:.2f}%")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Aleatório (AUC=0.5)")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (TAR / 1-FRR)")
    ax.set_title("Curva ROC — LFW e YTF (Baseline + Ataque Deepfake)\nAdaFace IR-50 + FaceFusion inswapper_128")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    # sem marcadores de threshold adicionais (já legível pela legenda)

    fig.tight_layout()
    path = os.path.join(OUT, "roc_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ ROC combinada → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAR / FRR VS THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────
def plot_far_frr(lfw_gen, lfw_imp, ytf_gen, ytf_imp):
    tau_range = np.arange(0.01, 0.99, 0.005)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, gen, imp, title, col_far, col_frr in [
        (axes[0], lfw_gen, lfw_imp, "LFW", "#E91E63", "#2196F3"),
        (axes[1], ytf_gen, ytf_imp, "YTF", "#FF9800", "#9C27B0"),
    ]:
        far, frr = far_frr_curve(gen, imp, tau_range)

        ax.plot(tau_range, far * 100, color=col_far, lw=2, label="FAR (False Accept Rate)")
        ax.plot(tau_range, frr * 100, color=col_frr, lw=2, label="FRR (False Reject Rate)")

        # EER
        idx_eer = np.argmin(np.abs(far - frr))
        eer_val = (far[idx_eer] + frr[idx_eer]) / 2 * 100
        ax.axvline(x=tau_range[idx_eer], color="black", lw=1.5, ls=":", alpha=0.8)
        ax.scatter([tau_range[idx_eer]], [eer_val], color="black", zorder=5, s=70)
        ax.annotate(f"EER={eer_val:.2f}%\nτ={tau_range[idx_eer]:.3f}",
                    xy=(tau_range[idx_eer], eer_val),
                    xytext=(tau_range[idx_eer] + 0.07, eer_val + 5),
                    fontsize=8.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"))

        for tau in THRESHOLDS:
            ax.axvline(x=tau, color="gray", lw=1, ls="--", alpha=0.6)
            ax.text(tau + 0.005, 95, f"τ={tau}", fontsize=7.5, color="gray")

        ax.set_xlabel("Threshold (τ)")
        ax.set_ylabel("Taxa de Erro (%)")
        ax.set_title(f"{title} — FAR e FRR vs Threshold\n(Baseline sem ataque)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 105)
        ax.set_xlim(0, 1)

    fig.suptitle("FAR / FRR vs Threshold — LFW e YTF\nAdaFace IR-50 (sem manipulação)", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT, "far_frr_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ FAR/FRR combinado → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CURVA DET (Detection Error Tradeoff)
# ─────────────────────────────────────────────────────────────────────────────
def probit(p):
    """Transformação probit (normal deviate)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.sqrt(2) * erfinv(2 * p - 1)


def plot_det(lfw_gen, lfw_imp, lfw_sw, ytf_gen, ytf_imp, ytf_sw):
    fig, ax = plt.subplots(figsize=(8, 7))

    configs = [
        (lfw_imp, lfw_gen, "LFW Baseline",         C_GEN,  "-"),
        (ytf_imp, ytf_gen, "YTF Baseline",          C_BASE, "-"),
        (lfw_imp, lfw_sw,  "LFW Ataque Deepfake",   C_LFW,  "--"),
        (ytf_imp, ytf_sw,  "YTF Ataque Deepfake",   C_YTF,  "--"),
    ]

    ticks = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    tick_labels = ["1%", "5%", "10%", "20%", "30%", "40%", "50%"]

    for neg, pos, label, color, ls in configs:
        y_true  = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        y_score = np.concatenate([pos, neg])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        # Filtra extremos
        mask = (fpr > 0) & (fpr < 1) & (fnr > 0) & (fnr < 1)
        ax.plot(probit(fpr[mask]), probit(fnr[mask]),
                color=color, lw=2, ls=ls, label=label)

    ax.set_xticks([probit(t) for t in ticks])
    ax.set_xticklabels(tick_labels)
    ax.set_yticks([probit(t) for t in ticks])
    ax.set_yticklabels(tick_labels)

    # Linha diagonal EER
    lim = probit(0.5)
    ax.plot([-4, lim], [-4, lim], "k--", lw=1, alpha=0.3, label="EER line")

    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("False Reject Rate (FRR)")
    ax.set_title("Curva DET — Detection Error Tradeoff\nLFW e YTF (Baseline + Ataque Deepfake)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(probit(0.01), probit(0.6))
    ax.set_ylim(probit(0.01), probit(0.6))

    fig.tight_layout()
    path = os.path.join(OUT, "det_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Curva DET → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ASR COMBINADA (LFW vs YTF)
# ─────────────────────────────────────────────────────────────────────────────
def plot_asr_combined(lfw_sw, ytf_sw):
    tau_range = np.arange(0.05, 0.95, 0.01)

    asr_lfw = [np.mean(lfw_sw >= t) * 100 for t in tau_range]
    asr_ytf = [np.mean(ytf_sw >= t) * 100 for t in tau_range]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tau_range, asr_lfw, color=C_LFW, lw=2.5, label=f"LFW — Fotos (n={len(lfw_sw):,})")
    ax.plot(tau_range, asr_ytf, color=C_YTF, lw=2.5, label=f"YTF — Vídeos (n={len(ytf_sw):,})")

    for tau in THRESHOLDS:
        asr_l = np.mean(lfw_sw >= tau) * 100
        asr_y = np.mean(ytf_sw >= tau) * 100
        ax.axvline(x=tau, color="black", lw=1.2, ls="--", alpha=0.6)
        ax.annotate(f"LFW: {asr_l:.1f}%\nYTF: {asr_y:.1f}%",
                    xy=(tau, (asr_l + asr_y) / 2),
                    xytext=(tau + 0.03, (asr_l + asr_y) / 2 + 3),
                    fontsize=8.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"))

    ax.fill_between(tau_range, asr_lfw, asr_ytf, alpha=0.1, color="gray",
                    label="Diferença LFW − YTF")

    ax.set_xlabel("Threshold (τ)")
    ax.set_ylabel("Attack Success Rate — ASR (%)")
    ax.set_title("ASR vs Threshold — LFW (Fotos) vs YTF (Vídeos)\nFaceFusion inswapper_128 + AdaFace IR-50")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 105)
    ax.set_xlim(0.05, 0.95)

    fig.tight_layout()
    path = os.path.join(OUT, "asr_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ ASR combinada → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HISTOGRAMA COMBINADO (4 distribuições)
# ─────────────────────────────────────────────────────────────────────────────
def plot_hist_combined(lfw_gen, lfw_imp, lfw_sw, ytf_sw):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, sw, title, c_sw in [
        (axes[0], lfw_sw, "LFW", C_LFW),
        (axes[1], ytf_sw, "YTF", C_YTF),
    ]:
        bins = np.linspace(-0.1, 1.0, 60)
        ax.hist(lfw_gen, bins=bins, alpha=0.55, density=True, color=C_GEN,
                label=f"Genuínos LFW (n={len(lfw_gen):,})")
        ax.hist(lfw_imp, bins=bins, alpha=0.55, density=True, color=C_IMP,
                label=f"Impostores LFW (n={len(lfw_imp):,})")
        ax.hist(sw, bins=bins, alpha=0.70, density=True, color=c_sw,
                label=f"Deepfakes {title} (n={len(sw):,})")

        for tau in THRESHOLDS:
            ax.axvline(x=tau, color="black", lw=1.5, ls="--", alpha=0.7)
            ax.text(tau + 0.01, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 5,
                    f"τ={tau}", fontsize=8, color="black")

        ax.set_xlabel("Similaridade Cosseno")
        ax.set_ylabel("Densidade")
        ax.set_title(f"{title} — Distribuição de Scores\n(Genuínos vs Impostores vs Deepfakes)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Impacto do Ataque Deepfake no Espaço de Scores Biométricos\n"
                 "AdaFace IR-50 — LFW (fotos) e YTF (vídeos)", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT, "hist_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Histograma combinado → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MATRIZ DE CONFUSÃO nos thresholds operacionais
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(lfw_gen, lfw_imp, ytf_gen, ytf_imp):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, (gen, imp, ds) in enumerate([
        (lfw_gen, lfw_imp, "LFW"),
        (ytf_gen, ytf_imp, "YTF"),
    ]):
        for col, tau in enumerate(THRESHOLDS):
            ax = axes[row][col]
            y_true  = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
            y_pred  = np.concatenate([(gen >= tau).astype(int), (imp >= tau).astype(int)])
            cm      = confusion_matrix(y_true, y_pred)

            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            acc   = (tp + tn) / total * 100
            far   = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
            frr   = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(f"{ds} — τ={tau}\nAcc={acc:.1f}% | FAR={far:.1f}% | FRR={frr:.1f}%")

            classes = ["Impostor\n(rejeitar)", "Genuíno\n(aceitar)"]
            ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
            ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")

            thresh = cm.max() / 2
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i,j]:,}\n({cm[i,j]/total*100:.1f}%)",
                            ha="center", va="center", fontsize=10,
                            color="white" if cm[i, j] > thresh else "black",
                            fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Matrizes de Confusão — LFW e YTF\n"
                 "Nos thresholds operacionais τ=0.45 e τ=0.60", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Matrizes de confusão → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. TABELA RESUMO
# ─────────────────────────────────────────────────────────────────────────────
def print_summary_table(lfw_gen, lfw_imp, lfw_sw, ytf_gen, ytf_imp, ytf_sw):
    print("\n" + "=" * 80)
    print(" TABELA RESUMO DE MÉTRICAS PARA O PAPER ".center(80, "═"))
    print("=" * 80)

    rows = []
    for gen, imp, sw, ds in [
        (lfw_gen, lfw_imp, lfw_sw, "LFW"),
        (ytf_gen, ytf_imp, ytf_sw, "YTF"),
    ]:
        y_true  = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
        y_score = np.concatenate([gen, imp])
        fpr, tpr, thr = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        eer, eer_thr = eer_from_roc(fpr, tpr, thr)

        for tau in THRESHOLDS:
            far = np.mean(imp >= tau) * 100
            frr = np.mean(gen <  tau) * 100
            asr = np.mean(sw  >= tau) * 100
            rows.append({
                "Dataset": ds,
                "τ": tau,
                "AUC": f"{roc_auc:.4f}",
                "EER (%)": f"{eer*100:.2f}",
                "EER τ": f"{eer_thr:.3f}",
                "FAR (%)": f"{far:.2f}",
                "FRR (%)": f"{frr:.2f}",
                "ASR (%)": f"{asr:.2f}",
                "n Swaps": len(sw),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    csv_path = os.path.join(OUT, "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  📄 Tabela salva em: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  GERANDO GRÁFICOS ACADÊMICOS PARA O PAPER".center(70))
    print("=" * 70)
    print(f"\n  Output → {OUT}\n")

    print("📂 Carregando dados...")
    lfw_gen, lfw_imp, lfw_sw, ytf_gen, ytf_imp, ytf_sw = load_data()
    print(f"  LFW: {len(lfw_gen):,} genuínos | {len(lfw_imp):,} impostores | {len(lfw_sw):,} swaps")
    print(f"  YTF: {len(ytf_gen):,} genuínos | {len(ytf_imp):,} impostores | {len(ytf_sw):,} swaps\n")

    print("📊 Gerando gráficos...")
    plot_roc_combined(lfw_gen, lfw_imp, lfw_sw, ytf_gen, ytf_imp, ytf_sw)
    plot_far_frr(lfw_gen, lfw_imp, ytf_gen, ytf_imp)
    plot_det(lfw_gen, lfw_imp, lfw_sw, ytf_gen, ytf_imp, ytf_sw)
    plot_asr_combined(lfw_sw, ytf_sw)
    plot_hist_combined(lfw_gen, lfw_imp, lfw_sw, ytf_sw)
    plot_confusion_matrices(lfw_gen, lfw_imp, ytf_gen, ytf_imp)
    print_summary_table(lfw_gen, lfw_imp, lfw_sw, ytf_gen, ytf_imp, ytf_sw)

    print(f"\n{'=' * 70}")
    print("  ✅ TODOS OS GRÁFICOS GERADOS!".center(70))
    print(f"  Pasta: {OUT}")
    print(f"{'=' * 70}")
    print("\n  Arquivos gerados:")
    for f in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"    📄 {f}  ({size/1024:.0f} KB)")
