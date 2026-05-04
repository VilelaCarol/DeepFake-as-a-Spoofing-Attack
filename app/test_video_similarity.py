"""
Script para comparar um vídeo com uma imagem de referência,
extrair as posições do rosto e desenhar a similaridade no frame.
"""
import os
import sys
import argparse
import numpy as np
import cv2

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--source-image", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="app/results/pose_analysis/frames_deepfake_anotados")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    verifier = init_adaface()
    
    # 1. Carregar imagem de referência
    img_ref = cv2.imread(args.source_image)
    if img_ref is None:
        print(f"Erro ao ler: {args.source_image}")
        return
        
    emb_ref_tensor = verifier.get_embedding(img_ref)
    if emb_ref_tensor is None:
        print("Nenhum rosto na imagem de referência!")
        return
    ref_emb = emb_ref_tensor.cpu().numpy()
    print("✅ Embedding de referência carregado!")
    
    # 2. Ler o vídeo deepfake
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎥 Processando {total_frames} frames do vídeo {args.video}...")
    
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
        
        emb_tensor = verifier.get_embedding(frame)
        sim = 0.0
        if emb_tensor is not None:
            emb = emb_tensor.cpu().numpy()
            sim = compute_similarity(emb, ref_emb)
            
        # --- DESENHAR NO FRAME ---
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
        text1 = f"Sim. (Atacante): {sim:.4f}"
        text2 = f"Pos: X={x_center:.2f}, Y={y_center:.2f}"
        
        cv2.putText(annotated_frame, text1, (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, text2, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frame_filename = os.path.join(args.out_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)
            
        if frame_idx % 30 == 0:
            print(f"   Salvando frame {frame_idx}/{total_frames}...")

    cap.release()
    print(f"\n✅ Todos os frames anotados foram salvos em: {args.out_dir}")

if __name__ == "__main__":
    main()
