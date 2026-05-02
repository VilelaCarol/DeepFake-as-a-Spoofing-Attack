"""
================================================================================
REAL-TIME DEEPFAKE API SERVER (WebSocket)
================================================================================
Servidor de inferência GPU — recebe frames JPEG do cliente (PC pessoal) via
WebSocket, executa face swap neural (inswapper_128 + ArcFace) e verificação
biométrica (AdaFace), e retorna o frame processado + métricas em tempo real.

Arquitetura:
  [PC Pessoal — Browser + Webcam]
        ↕  WebSocket (JPEG frames + JSON metrics)
  [Servidor GPU — RTX A5500]
        ↓
  [Face Detection: YOLOv8-Face]  →  [Face Swap: inswapper_128]
        ↓                                    ↓
  [Verificação Biométrica: AdaFace]  →  [Frame processado + Métricas]

Uso:
  python realtime_api_server.py --source-image data/dataset_pessoal/originais/victorsorrindoclaro.jpeg

  Acesse o client HTML no PC pessoal → ws://<IP_SERVIDOR>:8765
================================================================================
"""

import os
import sys

# ── Fix cuDNN: usa as libs bundled do pip (nvidia-cudnn-cu12) ─────────────────
# O sistema tem /usr/lib/libcudnn_cnn.so.9 com símbolo quebrado.
_CUDNN_LIB_PATH = None
for _sp in sys.path:
    _candidate = os.path.join(_sp, "nvidia", "cudnn", "lib")
    if os.path.isdir(_candidate):
        _CUDNN_LIB_PATH = _candidate
        break
if _CUDNN_LIB_PATH is None:
    _CUDNN_LIB_PATH = (
        "/home/victor/.local/lib/python3.13/site-packages/nvidia/cudnn/lib"
    )
if os.path.isdir(_CUDNN_LIB_PATH):
    _old = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = _CUDNN_LIB_PATH + (":" + _old if _old else "")
    print(f"🔧 cuDNN fix: {_CUDNN_LIB_PATH}")

import json
import time
import struct
import asyncio
import argparse
import csv

import cv2
import numpy as np
import torch

# ==============================================================================
# Path da API YOLOv8 + AdaFace
# ==============================================================================
YOLO8FACE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "yolo8face_adaface"
)
sys.path.insert(0, YOLO8FACE_DIR)

from src import AdaFaceVerifier

# ==============================================================================
# Configurações
# ==============================================================================
CONFIG_PATH   = os.path.join(YOLO8FACE_DIR, "configs", "config.yaml")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
THRESHOLD_LOW = 0.45
THRESHOLD_HI  = 0.60

# Cores BGR para o HUD
GREEN  = (0, 220, 80)
RED    = (0, 60, 220)
YELLOW = (0, 200, 220)
WHITE  = (255, 255, 255)
DARK   = (20, 20, 20)

# ==============================================================================
# Modelos globais (singleton)
# ==============================================================================
_verifier = None
_neural_swapper = None
_attacker_img = None
_ref_embedding = None
_yolo_detector = None


def init_models(source_image: str):
    """Inicializa todos os modelos na GPU."""
    global _verifier, _neural_swapper, _attacker_img, _ref_embedding

    print("=" * 60)
    print("🚀 REAL-TIME DEEPFAKE API SERVER")
    print("=" * 60)

    # 1. AdaFace Verifier
    print("\n🔧 Carregando AdaFaceVerifier (YOLOv8 + AdaFace CVLFace)...")
    old_cwd = os.getcwd()
    os.chdir(YOLO8FACE_DIR)
    try:
        _verifier = AdaFaceVerifier(CONFIG_PATH)
    finally:
        os.chdir(old_cwd)
    print(f"✅ AdaFace carregado em: {_verifier.device}")

    # 2. Imagem do atacante + embedding de referência
    _attacker_img = cv2.imread(source_image)
    if _attacker_img is None:
        raise FileNotFoundError(f"Imagem do atacante não encontrada: {source_image}")

    emb_tensor = _verifier.get_embedding(_attacker_img)
    if emb_tensor is None:
        raise RuntimeError(f"Nenhuma face detectada na imagem do atacante: {source_image}")
    _ref_embedding = emb_tensor.cpu().numpy()
    print(f"✅ Embedding do atacante extraído — shape: {_ref_embedding.shape}")

    # 3. Neural Face Swapper (inswapper_128 + ArcFace)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from neural_face_swap import NeuralFaceSwapper
    _neural_swapper = NeuralFaceSwapper(_attacker_img)
    print(f"✅ NeuralFaceSwapper carregado")

    # 4. Pre-load YOLO detector
    global _yolo_detector
    try:
        from src.utils import detector as yolo_det
        _yolo_detector = yolo_det
        # Teste rápido com a imagem do atacante
        test_results = _yolo_detector(_attacker_img, verbose=False)[0]
        print(f"✅ YOLO detector carregado — teste: {len(test_results.boxes)} face(s) encontrada(s)")
    except Exception as e:
        print(f"⚠️  Erro ao carregar YOLO detector: {e}")
        import traceback; traceback.print_exc()

    print(f"\n📷 Atacante: {source_image}")
    print("=" * 60)


def detect_face(frame_bgr: np.ndarray):
    """Detecção de face com YOLOv8-Face."""
    global _yolo_detector
    if _yolo_detector is None:
        return None
    try:
        results = _yolo_detector(frame_bgr, verbose=False)[0]
        if len(results.boxes) > 0:
            box = results.boxes.xyxy[0].cpu().numpy().astype(int)
            return tuple(box)  # (x1, y1, x2, y2)
    except Exception as e:
        print(f"⚠️  detect_face erro: {e}")
    return None


def draw_hud(frame: np.ndarray, sim: float, match: bool,
             fps: float, bbox, mode: str) -> np.ndarray:
    """Retorna o frame limpo sem nenhum texto ou quadrado para a chamada de vídeo."""
    return frame.copy()


_last_bbox = None
_bbox_frame_counter = 0

def process_frame(frame_bgr: np.ndarray, mode: str = "ATAQUE"):
    """
    Pipeline completo: detecta face → swap → verificação biométrica.
    Retorna: (frame_processado, métricas_dict)
    """
    global _last_bbox, _bbox_frame_counter
    
    # YOLO Tracker Otimizado: Só roda a detecção a cada 15 frames para maximizar a performance
    if _bbox_frame_counter % 15 == 0 or _last_bbox is None:
        bbox = detect_face(frame_bgr)
        if bbox is not None:
            _last_bbox = bbox
    else:
        bbox = _last_bbox
    _bbox_frame_counter += 1

    # Face Swap
    if mode == "ATAQUE" and bbox is not None:
        processed = _neural_swapper.swap(frame_bgr, bbox)
    else:
        processed = frame_bgr.copy()

    # Verificação biométrica ignorada para maximizar FPS no streaming
    sim = 0.0
    match = False

    metrics = {
        "similarity": round(sim, 6),
        "match_045": False,
        "match_060": False,
        "face_detected": bbox is not None,
        "mode": mode,
        "bbox": [int(x) for x in bbox] if bbox else None,
    }

    return processed, metrics


# ==============================================================================
# WebSocket Server
# ==============================================================================
async def handle_client(websocket):
    """Handler para cada cliente WebSocket conectado."""
    import websockets

    client_addr = websocket.remote_address
    print(f"\n🔌 Cliente conectado: {client_addr}")

    mode = "ATAQUE"
    frame_count = 0
    match_count = 0
    t0 = time.time()
    fps_counter = 0
    fps_val = 0.0
    fps_time = time.time()

    # CSV log
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "realtime_attack_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "mode", "similarity", "match_045", "match_060",
                         "elapsed_s", "fps"])

    try:
        async for message in websocket:
            # Mensagem de texto = comando de controle
            if isinstance(message, str):
                try:
                    cmd = json.loads(message)
                    if cmd.get("type") == "mode":
                        mode = cmd.get("value", "ATAQUE")
                        print(f"🔄 Modo alterado: {mode}")
                        await websocket.send(json.dumps({"type": "mode_ack", "mode": mode}))
                    elif cmd.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong", "ts": time.time()}))
                except json.JSONDecodeError:
                    pass
                continue

            # Mensagem binária = frame JPEG
            t_start = time.time()

            # Decodifica JPEG → BGR numpy
            jpg_array = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Redimensiona para 640×480 se necessário
            h, w = frame.shape[:2]
            if w > 800 or h > 600:
                frame = cv2.resize(frame, (640, 480))

            # Processa em thread separada para não bloquear o asyncio
            loop = asyncio.get_event_loop()
            processed, metrics = await loop.run_in_executor(
                None, process_frame, frame, mode
            )

            # FPS
            frame_count += 1
            fps_counter += 1
            if fps_counter >= 10:
                now = time.time()
                fps_val = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            if metrics["match_045"]:
                match_count += 1

            elapsed = time.time() - t0
            latency_ms = (time.time() - t_start) * 1000
            asr = 100.0 * match_count / frame_count if frame_count > 0 else 0.0

            # Desenha HUD no frame
            bbox = tuple(metrics["bbox"]) if metrics["bbox"] else None
            output = draw_hud(processed, metrics["similarity"],
                              metrics["match_045"], fps_val, bbox, mode)

            # Codifica frame processado como JPEG
            _, jpg_buf = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_bytes = jpg_buf.tobytes()

            # Monta resposta: JSON metrics + JPEG frame
            # Protocolo: [4 bytes tamanho JSON][JSON bytes][JPEG bytes]
            response_metrics = {
                **metrics,
                "frame": frame_count,
                "fps": round(fps_val, 1),
                "latency_ms": round(latency_ms, 1),
                "match_count": match_count,
                "total_frames": frame_count,
                "asr": round(asr, 2),
                "elapsed_s": round(elapsed, 1),
            }
            json_bytes = json.dumps(response_metrics).encode("utf-8")
            json_len = struct.pack("<I", len(json_bytes))
            response = json_len + json_bytes + jpg_bytes
            await websocket.send(response)

            # CSV log
            csv_writer.writerow([
                frame_count, mode, f"{metrics['similarity']:.6f}",
                int(metrics["match_045"]), int(metrics["match_060"]),
                f"{elapsed:.2f}", f"{fps_val:.1f}"
            ])

            # Log periódico
            if frame_count % 50 == 0:
                print(f"   [{elapsed:6.1f}s] frames={frame_count} | "
                      f"FPS={fps_val:.1f} | latency={latency_ms:.0f}ms | "
                      f"sim={metrics['similarity']:.4f} | ASR={asr:.1f}%")

    except Exception as e:
        if "closed" not in str(e).lower():
            print(f"⚠ Erro: {e}")
    finally:
        csv_file.close()
        elapsed = time.time() - t0
        asr = 100.0 * match_count / frame_count if frame_count > 0 else 0.0

        print(f"\n{'=' * 60}")
        print(f"📊 RELATÓRIO — Cliente {client_addr}")
        print(f"{'=' * 60}")
        print(f"   Duração          : {elapsed:.1f}s")
        print(f"   Frames processados: {frame_count}")
        print(f"   FPS médio        : {frame_count / elapsed:.1f}" if elapsed > 0 else "   FPS médio        : N/A")
        print(f"   Frames aceitos   : {match_count}")
        print(f"   ASR (tau=0.45)   : {asr:.2f}%")
        print(f"   Log salvo em     : {csv_path}")
        print(f"{'=' * 60}\n")


async def main_server(host: str, port: int):
    """Inicia o servidor WebSocket."""
    import websockets

    print(f"\n📡 WebSocket Server escutando em ws://{host}:{port}")
    print(f"   Abra o realtime_client.html no PC pessoal e conecte.")
    print(f"   Ctrl+C para encerrar.\n")

    async with websockets.serve(
        handle_client,
        host,
        port,
        max_size=10 * 1024 * 1024,  # 10 MB max por mensagem
        ping_interval=20,
        ping_timeout=60,
    ):
        await asyncio.Future()  # roda eternamente


# ==============================================================================
# Entry point
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-Time Deepfake API Server (WebSocket)"
    )
    p.add_argument(
        "--source-image", type=str,
        default="data/dataset_pessoal/originais/victorsorrindoclaro.jpeg",
        help="Imagem do atacante (rosto a injetar)"
    )
    p.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host para bind do servidor"
    )
    p.add_argument(
        "--port", type=int, default=8765,
        help="Porta WebSocket"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Inicializa modelos na GPU
    init_models(args.source_image)

    # Inicia servidor
    asyncio.run(main_server(args.host, args.port))
