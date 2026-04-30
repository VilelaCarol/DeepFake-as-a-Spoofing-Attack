"""
================================================================================
ETAPA 5 — Real-Time Deepfake Attack Pipeline (SEM SUDO / SEM SINK)
================================================================================
Pipeline de ataque em tempo real — totalmente self-contained, sem necessidade
de dispositivo v4l2loopback adicional nem de permissão sudo.

  [Celular via DroidCam OBS]
        ↓  OBS Virtual Camera  (/dev/video0)
  [Captura de frame ao vivo]
        ↓
  [Detecção de face: YOLOv8-Face]
        ↓
  [Face Swap: substitui rosto pelo atacante]
        ↓  frame manipulado (em memória)
  [Verificação Biométrica: AdaFace CVLFace IR-50]
        ↓
  [Resultado: Similaridade / ASR / Fraude detectada?]
        ↓
  [Preview local + CSV de resultados]

Requisitos (sem sudo):
  - OBS Studio com câmera virtual ativa → /dev/video0
  - pip install opencv-python torch numpy

Uso:
  python realtime_attack_pipeline.py \\
    --source 0 \\
    --source-image data/dataset_pessoal/originais/victorsorrindoclaro.jpeg

  Pressione 'q' para encerrar, 'm' para alternar ATAQUE <-> PASSTHROUGH.
================================================================================
"""

import os
import sys

# Detecta se há display disponível (SSH sem X forwarding)
# Precisa ser ANTES de importar cv2, senão o Qt crasha
_HAS_DISPLAY = bool(os.environ.get("DISPLAY"))
if not _HAS_DISPLAY:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import csv
import time
import argparse
import threading
import queue
import numpy as np
import torch

# ==============================================================================
# Path da API YOLOv8 + AdaFace (mesmo padrão do baseline_lfw.py)
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
# Singleton do verifier
# ==============================================================================
_verifier = None

def get_verifier() -> AdaFaceVerifier:
    global _verifier
    if _verifier is None:
        print("🚀 Carregando AdaFaceVerifier (YOLOv8 + AdaFace CVLFace)...")
        old_cwd = os.getcwd()
        os.chdir(YOLO8FACE_DIR)
        try:
            _verifier = AdaFaceVerifier(CONFIG_PATH)
        finally:
            os.chdir(old_cwd)
        print(f"✅ Modelo carregado em: {_verifier.device}")
    return _verifier


# ==============================================================================
# Carrega atacante e extrai embedding de referência
# ==============================================================================
def load_attacker(image_path: str, verifier: AdaFaceVerifier):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem do atacante não encontrada: {image_path}")
    emb_tensor = verifier.get_embedding(img)
    if emb_tensor is None:
        raise RuntimeError(f"Nenhuma face detectada na imagem do atacante: {image_path}")
    embedding = emb_tensor.cpu().numpy()
    print(f"✅ Embedding do atacante extraído — shape: {embedding.shape}")
    return img, embedding


# ==============================================================================
# Detecção de face (retorna bounding box ou None)
# ==============================================================================
def detect_face(verifier: AdaFaceVerifier, frame_bgr: np.ndarray):
    try:
        from src.utils import detector as yolo_det
        results = yolo_det(frame_bgr, verbose=False)[0]
        if len(results.boxes) > 0:
            box = results.boxes.xyxy[0].cpu().numpy().astype(int)
            return tuple(box)  # (x1, y1, x2, y2)
    except Exception:
        pass
    return None


# ==============================================================================
# Face Swap em memória — blending suave com seamlessClone
# ==============================================================================
_attacker_face_crop = None  # Cache do recorte facial do atacante

def _get_attacker_face(attacker_img: np.ndarray, verifier) -> np.ndarray:
    """Recorta apenas o rosto da imagem do atacante para swap mais limpo."""
    global _attacker_face_crop
    if _attacker_face_crop is not None:
        return _attacker_face_crop

    bbox = detect_face(verifier, attacker_img)
    if bbox is None:
        _attacker_face_crop = attacker_img
        return attacker_img

    x1, y1, x2, y2 = bbox
    h_img, w_img = attacker_img.shape[:2]
    # Margem extra para incluir testa/queixo
    margin_x = int((x2 - x1) * 0.15)
    margin_y = int((y2 - y1) * 0.20)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w_img, x2 + margin_x)
    y2 = min(h_img, y2 + margin_y)
    _attacker_face_crop = attacker_img[y1:y2, x1:x2].copy()
    return _attacker_face_crop


def swap_face(frame: np.ndarray, attacker_img: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Substitui a região da face detectada pelo rosto do atacante.
    Usa seamlessClone para transição suave + correção de cor.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 10 or h <= 10:
        return frame

    fh, fw = frame.shape[:2]

    # Adiciona margem para seamlessClone funcionar longe das bordas
    margin = 10
    x1c = max(margin, x1)
    y1c = max(margin, y1)
    x2c = min(fw - margin, x2)
    y2c = min(fh - margin, y2)
    wc, hc = x2c - x1c, y2c - y1c
    if wc <= 10 or hc <= 10:
        return frame

    src_resized = cv2.resize(attacker_img, (wc, hc))

    # Correção de cor — ajusta tom de pele do atacante ao alvo
    target_roi = frame[y1c:y2c, x1c:x2c]
    src_lab = cv2.cvtColor(src_resized, cv2.COLOR_BGR2LAB).astype(float)
    tgt_lab = cv2.cvtColor(target_roi, cv2.COLOR_BGR2LAB).astype(float)
    for ch in range(3):
        src_mean, src_std = src_lab[:,:,ch].mean(), src_lab[:,:,ch].std() + 1e-6
        tgt_mean, tgt_std = tgt_lab[:,:,ch].mean(), tgt_lab[:,:,ch].std() + 1e-6
        src_lab[:,:,ch] = (src_lab[:,:,ch] - src_mean) * (tgt_std / src_std) + tgt_mean
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    src_corrected = cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

    # Máscara elíptica suave
    mask = np.zeros((hc, wc), dtype=np.uint8)
    cx, cy = wc // 2, hc // 2
    cv2.ellipse(mask, (cx, cy), (max(cx - 8, 1), max(cy - 8, 1)), 0, 0, 360, 255, -1)
    # Suaviza bordas da máscara
    mask = cv2.GaussianBlur(mask, (21, 21), 11)

    output = frame.copy()
    try:
        center = (x1c + cx, y1c + cy)
        output = cv2.seamlessClone(src_corrected, output, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # Fallback: blending Gaussiano manual
        mask_f = mask.astype(float) / 255.0
        mask_3c = np.stack([mask_f, mask_f, mask_f], axis=-1)
        roi = output[y1c:y2c, x1c:x2c].astype(float)
        blended = src_corrected.astype(float) * mask_3c + roi * (1 - mask_3c)
        output[y1c:y2c, x1c:x2c] = blended.astype(np.uint8)
    return output



# ==============================================================================
# HUD
# ==============================================================================
def draw_hud(frame: np.ndarray, sim: float, match: bool,
             fps: float, bbox, mode: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if bbox is not None:
        color = RED if match else GREEN
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # Painel superior
    cv2.rectangle(out, (0, 0), (w, 55), DARK, -1)
    cv2.putText(out, f"MODE: {mode}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    # Painel inferior
    cv2.rectangle(out, (0, h - 65), (w, h), DARK, -1)
    col = RED if match else GREEN
    tau_low = "PASS" if sim >= THRESHOLD_LOW else "FAIL"
    tau_hi  = "PASS" if sim >= THRESHOLD_HI  else "FAIL"
    verdict = "🚨 FRAUDE ACEITA  (ASR +1)" if match else "🛡  BLOQUEADO"

    cv2.putText(out,
                f"Sim: {sim:.4f}  |  tau=0.45: {tau_low}  |  tau=0.60: {tau_hi}",
                (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(out, verdict,
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    return out


# ==============================================================================
# Servidor MJPEG HTTP simples (para visualização remota sem ffmpeg pipe duplo)
# Uso: ffplay http://IP_DO_VICTOR:8080/  ou abrir no browser
# ==============================================================================
import io
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

class MJPEGServer:
    """Servidor HTTP que transmite frames como MJPEG stream."""

    def __init__(self, host="0.0.0.0", port=8080):
        self.host  = host
        self.port  = port
        self._lock = threading.Lock()
        self._frame_jpg: bytes = b""
        self._server = None
        self._thread = None

    def push_frame(self, frame_bgr: np.ndarray, quality: int = 70):
        """Atualiza o frame atual (thread-safe)."""
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self._lock:
                self._frame_jpg = buf.tobytes()

    def get_frame(self) -> bytes:
        with self._lock:
            return self._frame_jpg

    def start(self):
        server = self  # referência para o handler

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass   # silencia logs HTTP

            def do_GET(self):
                if self.path not in ("/", "/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=--frame")
                self.end_headers()
                try:
                    while True:
                        jpg = server.get_frame()
                        if jpg:
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + jpg + b"\r\n"
                            )
                        time.sleep(0.033)   # ~30 fps
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = HTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        print(f"📡 Servidor MJPEG iniciado → http://{self.host}:{self.port}/")
        print(f"   Na Ubuntu rode: ffplay http://192.168.18.254:{self.port}/")

    def stop(self):
        if self._server:
            self._server.shutdown()



class BiometricWorker(threading.Thread):
    def __init__(self, verifier: AdaFaceVerifier, ref_embedding: np.ndarray):
        super().__init__(daemon=True)
        self.verifier      = verifier
        self.ref_embedding = ref_embedding
        self.q             = queue.Queue(maxsize=2)
        self.last_sim      = 0.0
        self.last_match    = False
        self._stop         = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                frame = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                emb_tensor = self.verifier.get_embedding(frame)
                if emb_tensor is not None:
                    emb = emb_tensor.cpu().numpy()
                    sim = float(np.dot(emb, self.ref_embedding) /
                                (np.linalg.norm(emb) * np.linalg.norm(self.ref_embedding) + 1e-8))
                    self.last_sim   = sim
                    self.last_match = sim >= THRESHOLD_LOW
            except Exception:
                pass

    def enqueue(self, frame: np.ndarray):
        if not self.q.full():
            self.q.put(frame.copy())

    def stop(self):
        self._stop.set()


# ==============================================================================
# Captura via ffmpeg subprocess pipe (para streams UDP/RTSP)
# Bypass total do backend de rede do OpenCV
# ==============================================================================
def open_ffmpeg_pipe(url: str, width: int = 640, height: int = 480):
    """Abre stream de rede via ffmpeg e retorna processo com frames raw no stdout."""
    import subprocess

    # Se for TCP, adiciona ?listen para o ffmpeg atuar como servidor
    input_url = url
    if url.startswith("tcp://") and "listen" not in url:
        input_url = url + "?listen"

    cmd = [
        "ffmpeg",
        "-y",
        "-fflags", "+nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-probesize", "5000000",
        "-analyzeduration", "5000000",
        "-i", input_url,
        "-vf", f"scale={width}:{height}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-loglevel", "warning",
        "pipe:1"                           # ← corrigido: era "-" que não funciona no Arch
    ]
    print(f"   [DEBUG] ffmpeg cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=width * height * 3
    )
    return proc


def open_ffmpeg_output_pipe(url: str, width: int = 640, height: int = 480, fps: int = 30):
    """Abre pipe do ffmpeg para transmitir frames manipulados via rede."""
    import subprocess
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",               # Entrada via pipe (stdin)
        "-vcodec", "mjpeg",      # MJPEG é rápido para live
        "-q:v", "5",
        "-f", "mpegts",
        url
    ]
    print(f"   [DEBUG-OUT] ffmpeg output cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    return proc


def read_ffmpeg_frame(proc, width: int = 640, height: int = 480):
    """Lê um frame BGR do pipe ffmpeg. Retorna None se fim ou erro."""
    frame_bytes = width * height * 3
    raw = proc.stdout.read(frame_bytes)
    if len(raw) < frame_bytes:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))


# ==============================================================================
# Loop principal
# ==============================================================================
def run_pipeline(source, source_image: str, mode: str, output_url: str = None):
    """
    source: int (índice /dev/videoX) ou str (URL: udp://, rtsp://, http://)
    """
    print("\n" + "=" * 60)
    print("🎬  REAL-TIME DEEPFAKE ATTACK PIPELINE — Etapa 5")
    print("    Leitura inline — sem sudo, sem sink device")
    print("=" * 60)

    verifier = get_verifier()
    attacker_img, ref_emb = load_attacker(source_image, verifier)

    is_url = isinstance(source, str) and "://" in source
    FRAME_W, FRAME_H = 640, 480

    print(f"\n📷 Abrindo fonte: {source}")

    if is_url:
        # ── Stream de rede: usa ffmpeg subprocess pipe ──────────────
        print("   Modo: ffmpeg pipe (UDP/RTSP)")
        print(f"   ⏳ Conectando a {source} ...")
        print("   ⚠  Inicie o ffmpeg na Ubuntu ANTES de rodar este script!")
        proc = open_ffmpeg_pipe(source, FRAME_W, FRAME_H)
        cap  = None
        fps  = 30

    else:
        # ── Device local: usa OpenCV V4L2 ────────────────────────────
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        proc = None
        if not cap.isOpened():
            raise RuntimeError(
                f"❌ Não foi possível abrir /dev/video{source}.\n"
                "   Verifique se o OBS Virtual Camera está ativo."
            )
        FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"   Resolução: {FRAME_W}x{FRAME_H} @ {fps}fps")

    # Inicializa face swap neural (mesmo modelo do FaceFusion)
    from neural_face_swap import NeuralFaceSwapper
    neural_swapper = NeuralFaceSwapper(attacker_img)

    worker = BiometricWorker(verifier, ref_emb)
    worker.start()

    print(f"\n🔴 Pipeline iniciado")
    print(f"   Fonte    : {source}")
    print(f"   Atacante : {source_image}")
    print(f"   Modo     : {mode}")
    print(f"   Teclas   : 'q' = sair | 'm' = alternar ATAQUE/PASSTHROUGH\n")

    stats = {"total": 0, "match": 0, "t0": time.time()}
    fps_c, fps_val, fps_t = 0, 0.0, time.time()
    headless = not _HAS_DISPLAY
    if headless:
        print("📺 Modo headless (sem display) — output via servidor MJPEG HTTP")
    vid_out    = None
    out_proc   = None
    mjpeg_srv  = None

    # Inicia servidor MJPEG HTTP (sempre no modo headless ou se --output-url for porta)
    mjpeg_port = 8080
    if output_url and output_url.isdigit():
        mjpeg_port = int(output_url)
    mjpeg_srv = MJPEGServer(port=mjpeg_port)
    mjpeg_srv.start()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "realtime_attack_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "mode", "similarity", "match_045", "match_060",
                         "elapsed_s", "fps"])

    try:
        reconnect_attempts = 0
        MAX_RECONNECTS = 120  # tenta por até 120s (1 tentativa/s)

        while True:
            # ── Lê frame da fonte correta ─────────────────────────────
            if is_url:
                frame = read_ffmpeg_frame(proc, FRAME_W, FRAME_H)
                if frame is None:
                    reconnect_attempts += 1
                    if reconnect_attempts > MAX_RECONNECTS:
                        # Dump ffmpeg stderr for diagnosis
                        try:
                            err = proc.stderr.read().decode(errors='replace')
                            if err.strip():
                                print(f"   [ffmpeg stderr]: {err[-500:]}")
                        except Exception:
                            pass
                        print("❌ Stream não recebido após 120s. Encerrando.")
                        break
                    # Tenta reconectar
                    if reconnect_attempts == 1:
                        print(f"⏳ Aguardando stream UDP em {source}...")
                        print("   (Inicie o ffmpeg na Ubuntu se ainda não o fez)")
                    elif reconnect_attempts % 10 == 0:
                        print(f"   ... tentativa {reconnect_attempts}/{MAX_RECONNECTS}")
                        # Mostra stderr parcial do ffmpeg
                        try:
                            err = proc.stderr.read(2048)
                            if err:
                                print(f"   [ffmpeg]: {err.decode(errors='replace')[-200:]}")
                        except Exception:
                            pass
                    proc.kill()
                    proc = open_ffmpeg_pipe(source, FRAME_W, FRAME_H)
                    time.sleep(1)
                    continue
                # Stream voltou — reseta contador
                reconnect_attempts = 0
            else:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            stats["total"] += 1
            fps_c += 1
            if fps_c >= 30:
                fps_val = fps_c / (time.time() - fps_t)
                fps_c, fps_t = 0, time.time()

            bbox = detect_face(verifier, frame)

            if mode == "ATAQUE" and bbox is not None:
                processed = neural_swapper.swap(frame, bbox)
            else:
                processed = frame.copy()

            worker.enqueue(processed)

            sim   = worker.last_sim
            match = worker.last_match
            if match:
                stats["match"] += 1

            output = draw_hud(processed, sim, match, fps_val, bbox, mode)

            # Envia frame ao servidor MJPEG (visualização ao vivo na Ubuntu)
            mjpeg_srv.push_frame(output)

            # Exibe janela local se houver display
            if not headless:
                try:
                    cv2.imshow("Real-Time Deepfake Attack Pipeline", output)
                except cv2.error:
                    headless = True

            elapsed = time.time() - stats["t0"]
            csv_writer.writerow([
                stats["total"], mode, f"{sim:.6f}",
                int(sim >= THRESHOLD_LOW), int(sim >= THRESHOLD_HI),
                f"{elapsed:.2f}", f"{fps_val:.1f}"
            ])

            if stats["total"] % (fps * 5) == 0 and stats["total"] > 0:
                asr = 100.0 * stats["match"] / stats["total"]
                print(f"   [{elapsed:6.1f}s] frames={stats['total']} | "
                      f"FPS={fps_val:.1f} | sim={sim:.4f} | ASR={asr:.1f}%")

            if not headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    mode = "PASSTHROUGH" if mode == "ATAQUE" else "ATAQUE"
                    print(f"🔄 Modo: {mode}")

    except KeyboardInterrupt:
        print("\n⛔ Interrompido.")
    finally:
        worker.stop()
        if cap:
            cap.release()
        if proc:
            proc.kill()
        if out_proc:
            out_proc.stdin.close()
            out_proc.kill()
        if mjpeg_srv:
            mjpeg_srv.stop()
        csv_file.close()
        cv2.destroyAllWindows()

        elapsed = time.time() - stats["t0"]
        asr = 100.0 * stats["match"] / stats["total"] if stats["total"] else 0.0

        print("\n" + "=" * 60)
        print("📊 RELATÓRIO FINAL — ATAQUE EM TEMPO REAL")
        print("=" * 60)
        print(f"   Duração          : {elapsed:.1f}s")
        print(f"   Frames capturados: {stats['total']}")
        print(f"   FPS médio        : {stats['total'] / elapsed:.1f}")
        print(f"   Frames aceitos   : {stats['match']}")
        print(f"   ASR (tau=0.45)   : {asr:.2f}%")
        print(f"   Log salvo em     : {csv_path}")
        print("=" * 60)


# ==============================================================================
# Entry point
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-Time Deepfake Attack Pipeline — Etapa 5 (sem sudo)"
    )
    p.add_argument(
        "--source", type=str, default="0",
        help=(
            "Fonte de vídeo. Pode ser:\n"
            "  Device local : 0  (= /dev/video0, OBS Virtual Camera)\n"
            "  UDP stream   : udp://0.0.0.0:1234\n"
            "  RTSP stream  : rtsp://192.168.x.x:8554/live\n"
            "  HTTP/MJPEG   : http://192.168.x.x:8080/video"
        )
    )
    p.add_argument(
        "--source-image", type=str,
        default="data/dataset_pessoal/originais/victorsorrindoclaro.jpeg",
        help="Imagem do atacante (rosto a injetar)"
    )
    p.add_argument(
        "--mode", type=str, default="ATAQUE",
        choices=["ATAQUE", "PASSTHROUGH"],
        help="ATAQUE: injeta rosto manipulado | PASSTHROUGH: passa frame limpo"
    )
    p.add_argument(
        "--output-url", type=str, default=None,
        help="URL para transmitir o resultado (ex: udp://192.168.18.86:1235)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Converte para int se for número puro (device index)
    source = int(args.source) if args.source.isdigit() else args.source
    run_pipeline(
        source       = source,
        source_image = args.source_image,
        mode         = args.mode,
        output_url   = args.output_url
    )
