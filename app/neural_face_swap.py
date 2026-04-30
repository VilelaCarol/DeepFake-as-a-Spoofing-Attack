"""
Neural Face Swap — inswapper_128 (mesma tecnologia do FaceFusion)
================================================================
Usa os modelos ONNX do FaceFusion diretamente via onnxruntime-gpu.

FIXES aplicados:
  - LD_LIBRARY_PATH aponta para cuDNN bundled do pip (evita cuDNN quebrado do sistema)
  - Preprocessing correto do inswapper: BGR→float32/255, CHW, sem normalização extra
  - ArcFace: BGR, normalizado [-1,1]
"""

import os
import sys

# ── Fix cuDNN: usa as libs bundled do pip (nvidia-cudnn-cu12) ─────────────────
# O sistema tem /usr/lib/libcudnn_cnn.so.9 com símbolo quebrado.
# O pip instalou a versão correta em site-packages/nvidia/cudnn/lib/
_CUDNN_LIB_PATH = None
for _sp in sys.path:
    _candidate = os.path.join(_sp, "nvidia", "cudnn", "lib")
    if os.path.isdir(_candidate):
        _CUDNN_LIB_PATH = _candidate
        break
# Tenta o caminho fixo (Python 3.13 system)
if _CUDNN_LIB_PATH is None:
    _CUDNN_LIB_PATH = (
        "/home/victor/.local/lib/python3.13/site-packages/nvidia/cudnn/lib"
    )

if os.path.isdir(_CUDNN_LIB_PATH):
    _old = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = _CUDNN_LIB_PATH + (":" + _old if _old else "")

# ── Agora importa onnxruntime (já com LD_LIBRARY_PATH corrigido) ──────────────
import cv2
import numpy as np
import onnxruntime as ort
import onnx
import onnx.numpy_helper as numpy_helper


# Caminhos padrão dos modelos (FaceFusion)
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "facefusion", ".assets", "models"
)
_INSWAPPER = os.path.join(_MODELS_DIR, "inswapper_128.onnx")
_ARCFACE   = os.path.join(_MODELS_DIR, "arcface_w600k_r50.onnx")


class NeuralFaceSwapper:
    """Face swap neural usando inswapper_128 + ArcFace (GPU via CUDA/TensorRT)."""

    def __init__(self, source_image: np.ndarray, source_bbox: tuple = None):
        """
        source_image : imagem BGR do atacante
        source_bbox  : (x1,y1,x2,y2) opcional — se None usa a imagem inteira
        """
        # Tenta GPU (CUDA) → TensorRT → CPU
        providers = self._build_providers()

        print("🧠 Carregando modelos neurais de face swap...")
        print(f"   Providers disponíveis: {ort.get_available_providers()}")
        self.arcface = ort.InferenceSession(_ARCFACE, providers=providers)
        self.swapper = ort.InferenceSession(_INSWAPPER, providers=providers)
        print(f"   ArcFace rodando em : {self.arcface.get_providers()}")
        print(f"   Inswapper rodando em: {self.swapper.get_providers()}")

        # Inicializador do inswapper necessário para converter o embedding do ArcFace
        inswapper_model = onnx.load(_INSWAPPER)
        self.swapper_initializer = numpy_helper.to_array(inswapper_model.graph.initializer[-1])

        self.source_embedding = self._get_arcface_embedding(source_image, source_bbox)
        print(f"   ✅ Embedding do atacante — shape: {self.source_embedding.shape}")

    @staticmethod
    def _build_providers():
        """Escolhe o melhor provider disponível."""
        available = ort.get_available_providers()
        for p in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
            if p in available:
                return [p, "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # ── Alinhamento ──────────────────────────────────────────────────────────
    def _crop_face(self, image: np.ndarray, bbox: tuple,
                   size: int, scale: float = 1.5) -> np.ndarray:
        """Recorta região da face com margem e redimensiona para `size`."""
        x1, y1, x2, y2 = bbox
        h_img, w_img = image.shape[:2]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = int(max(w, h) * scale / 2)
        x1n = max(0, cx - half)
        y1n = max(0, cy - half)
        x2n = min(w_img, cx + half)
        y2n = min(h_img, cy + half)
        crop = image[y1n:y2n, x1n:x2n]
        return cv2.resize(crop, (size, size)), (x1n, y1n, x2n, y2n)

    # ── Preprocessing ────────────────────────────────────────────────────────
    def _preprocess_arcface(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        ArcFace espera: RGB float32 normalizado [-1, 1], shape [1,3,112,112].
        """
        face_rgb = face_bgr[:, :, ::-1]  # BGR → RGB
        face = face_rgb.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        face = face.transpose(2, 0, 1)                     # HWC → CHW
        return np.expand_dims(face, 0)                     # [1,3,112,112]

    def _preprocess_swap(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Inswapper espera: RGB float32 / 255, shape [1,3,128,128].
        """
        face_rgb = face_bgr[:, :, ::-1]  # BGR → RGB
        face = face_rgb.astype(np.float32) / 255.0
        face = face.transpose(2, 0, 1)
        return np.expand_dims(face, 0)                     # [1,3,128,128]

    def _postprocess_swap(self, tensor: np.ndarray) -> np.ndarray:
        """Tensor [1,3,128,128] → imagem BGR uint8."""
        img = tensor[0].transpose(1, 2, 0)                 # CHW → HWC (RGB)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = img[:, :, ::-1]  # RGB → BGR
        return img                                         # BGR

    # ── ArcFace embedding ────────────────────────────────────────────────────
    def _get_arcface_embedding(self, image: np.ndarray,
                               bbox: tuple = None) -> np.ndarray:
        if bbox is None:
            face_112 = cv2.resize(image, (112, 112))
        else:
            face_112, _ = self._crop_face(image, bbox, 112)

        tensor = self._preprocess_arcface(face_112)
        emb = self.arcface.run(None, {"input": tensor})[0].flatten()
        
        # Multiplica pelo inicializador do Inswapper e normaliza novamente!
        emb = np.dot(emb.reshape(1, -1), self.swapper_initializer)
        emb = emb / np.linalg.norm(emb)
        return emb.flatten()

    # ── Swap principal ───────────────────────────────────────────────────────
    def swap(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Aplica swap neural no frame usando inswapper_128.
        bbox: (x1, y1, x2, y2) da face detectada.
        """
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return frame

        # 1. Recorta face alvo (vítima) → 128x128
        face_128, crop_box = self._crop_face(frame, bbox, 128)
        target_tensor = self._preprocess_swap(face_128)

        # 2. Inswapper: injeta rosto do atacante
        src_emb = self.source_embedding.reshape(1, 512).astype(np.float32)
        swapped_tensor = self.swapper.run(None, {
            "target": target_tensor,
            "source": src_emb
        })[0]

        # 3. Pós-processamento → BGR uint8
        swapped_128 = self._postprocess_swap(swapped_tensor)

        # 4. Redimensiona de volta para o tamanho do crop original
        cx1, cy1, cx2, cy2 = crop_box
        crop_h, crop_w = cy2 - cy1, cx2 - cx1
        swapped_full = cv2.resize(swapped_128, (crop_w, crop_h))

        # 5. Máscara elíptica suave para blending
        output = frame.copy()
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        mcx, mcy = crop_w // 2, crop_h // 2
        cv2.ellipse(mask, (mcx, mcy),
                    (max(mcx - 10, 1), max(mcy - 10, 1)),
                    0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 15)

        # 6. seamlessClone (mesclagem perfeita de bordas)
        fh, fw = frame.shape[:2]
        cx = max(15, min(fw - 15, cx1 + mcx))
        cy_c = max(15, min(fh - 15, cy1 + mcy))
        try:
            output = cv2.seamlessClone(
                swapped_full, output, mask, (cx, cy_c), cv2.NORMAL_CLONE
            )
        except cv2.error:
            # Fallback: alpha blending suave
            mask_f = mask.astype(float) / 255.0
            mask_3c = np.stack([mask_f] * 3, axis=-1)
            roi = output[cy1:cy2, cx1:cx2].astype(float)
            blended = swapped_full.astype(float) * mask_3c + roi * (1 - mask_3c)
            output[cy1:cy2, cx1:cx2] = blended.astype(np.uint8)

        return output
