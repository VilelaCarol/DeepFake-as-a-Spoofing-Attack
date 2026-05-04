"""
Análise Estatística da Influência da Posição do Rosto na Similaridade Biométrica.

Lê um vídeo ("videojoao.mp4"), extrai frames, detecta a posição (X, Y) e 
tamanho (Área) do rosto usando YOLOv8-face.
Extrai o embedding com AdaFace e calcula a similaridade contra o primeiro frame.
Salva os frames anotados com a similaridade, gera um log detalhado e faz a análise
de significância da hipótese.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

YOLO8FACE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../yolo8face_adaface"
)
sys.path.insert(0, YOLO8FACE_DIR)
from src import AdaFaceVerifier
from src.utils import detector as yolo_detector

CONFIG_PATH = os.path.join(YOLO8FACE_DIR, "configs", "config.yaml")

def init_adaface():
    print("🚀 Inicializando AdaFaceVerifier...")
    cwd = os.getcwd()
    os.chdir(YOLO8FACE_DIR)
    verifier = AdaFaceVerifier(CONFIG_PATH)
    os.chdir(cwd)
    return verifier

def compute_similarity(emb_a, emb_b):
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

def main():
    parser = argparse.ArgumentParser(description="Análise de Posição vs Similaridade")
    parser.add_argument("--video", type=str, required=True,
                        help="Caminho do vídeo a ser analisado")
    parser.add_argument("--out-dir", type=str, default="app/results/pose_analysis",
                        help="Pasta para salvar os resultados")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"❌ ERRO: Vídeo não encontrado em {video_path}")
        return

    frames_out_dir = os.path.join(args.out_dir, "frames_anotados")
    os.makedirs(frames_out_dir, exist_ok=True)

    verifier = init_adaface()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n🎥 Processando vídeo: {video_path}")
    print(f"   Total Frames: {total_frames} | FPS: {fps:.2f}")

    ref_emb = None
    results_data = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        h, w, _ = frame.shape
        
        yolo_res = yolo_detector(frame, verbose=False)[0]
        if len(yolo_res.boxes) == 0:
            continue
            
        box = yolo_res.boxes.xyxy[0].cpu().numpy().astype(int)
        
        # Coordenadas relativas
        x_center = ((box[0] + box[2]) / 2) / w
        y_center = ((box[1] + box[3]) / 2) / h
        
        face_width = (box[2] - box[0]) / w
        face_height = (box[3] - box[1]) / h
        face_area = face_width * face_height

        emb_tensor = verifier.get_embedding(frame)
        if emb_tensor is None:
            continue
            
        emb = emb_tensor.cpu().numpy()
        
        if ref_emb is None:
            ref_emb = emb
            print(f"✅ Frame de referência capturado (Frame {frame_idx})")
            continue
            
        sim = compute_similarity(emb, ref_emb)
        
        results_data.append({
            "frame": frame_idx,
            "x_center": x_center,
            "y_center": y_center,
            "face_area": face_area,
            "similarity": sim
        })
        
        # --- DESENHAR NO FRAME ---
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        text1 = f"Similaridade: {sim:.4f}"
        text2 = f"Pos: X={x_center:.2f}, Y={y_center:.2f}"
        
        cv2.putText(annotated_frame, text1, (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, text2, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frame_filename = os.path.join(frames_out_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)
        
        if frame_idx % 30 == 0:
            print(f"   Processando frame {frame_idx}/{total_frames}...")

    cap.release()
    
    if not results_data:
        print("❌ Nenhum dado coletado. Rosto não foi detectado no vídeo.")
        return

    df = pd.DataFrame(results_data)
    
    # Salvar log em TXT detalhado
    txt_log_path = os.path.join(args.out_dir, "resultados_frames.txt")
    with open(txt_log_path, "w") as f:
        f.write("LOG DE RESULTADOS POR FRAME\n")
        f.write("="*60 + "\n")
        f.write("Frame | Sim. Cosseno | Posição X | Posição Y | Área Relativa\n")
        f.write("-" * 60 + "\n")
        for row in results_data:
            f.write(f"{row['frame']:04d}  | {row['similarity']:.4f}       | {row['x_center']:.4f}    | {row['y_center']:.4f}    | {row['face_area']:.4f}\n")
    
    print(f"\n✅ Log detalhado salvo em: {txt_log_path}")
    print(f"✅ Frames anotados salvos em: {frames_out_dir}")

    # =========================================================================
    # ANÁLISE ESTATÍSTICA (TESTE DE HIPÓTESES SIMPLIFICADO)
    # =========================================================================
    print("\n" + "="*60)
    print("🔬 TESTE DE HIPÓTESES: A POSIÇÃO AFETA A SIMILARIDADE?")
    print("="*60)
    
    report = []
    
    def test_hypothesis(var_name, var_col):
        valid_df = df.dropna(subset=[var_col, "similarity"])
        if len(valid_df) < 3:
            return f"\n🔹 Hipótese sobre {var_name}: Dados insuficientes."
            
        corr, p_value = stats.pearsonr(valid_df[var_col], valid_df["similarity"])
        
        # Lógica de significância baseada no p-value
        if p_value < 0.001:
            conclusao = "EXTREMAMENTE SIGNIFICATIVO (Certeza estatística quase total >99.9%)"
            chance = "Muito Alta"
        elif p_value < 0.05:
            conclusao = "ESTATISTICAMENTE SIGNIFICATIVO (Confiança >95%)"
            chance = "Alta"
        else:
            conclusao = "NÃO SIGNIFICATIVO (Não é possível provar a hipótese)"
            chance = "Baixa"
            
        # Explicação da Correlação de Pearson
        if corr < 0:
            efeito = f"NEGATIVO (r = {corr:.4f}). Isso significa que quanto maior o {var_name}, MENOR é a similaridade."
        else:
            efeito = f"POSITIVO (r = {corr:.4f}). Isso significa que quanto maior o {var_name}, MAIOR é a similaridade."
            
        rep = f"\n🔹 Hipótese: O(a) {var_name} afeta a Similaridade Biométrica?\n"
        rep += f"   - Correlação de Pearson: {efeito}\n"
        rep += f"   - Valor-p: {p_value:.4e}\n"
        rep += f"   - Chance da Hipótese estar certa: {chance}\n"
        rep += f"   - Conclusão: O efeito é {conclusao}.\n"
        return rep

    df["x_deviation"] = abs(df["x_center"] - 0.5)
    df["y_deviation"] = abs(df["y_center"] - 0.5)

    report.append(test_hypothesis("Desvio Horizontal (Sair do centro para os lados)", "x_deviation"))
    report.append(test_hypothesis("Desvio Vertical (Mover o rosto para cima/baixo)", "y_deviation"))
    report.append(test_hypothesis("Tamanho do Rosto (Distância da câmera)", "face_area"))
    
    report_text = "".join(report)
    print(report_text)
    
    with open(os.path.join(args.out_dir, "statistical_report.txt"), "w") as f:
        f.write("RELATÓRIO ESTATÍSTICO DA HIPÓTESE\n")
        f.write("="*50 + "\n")
        f.write(report_text)

if __name__ == "__main__":
    main()
