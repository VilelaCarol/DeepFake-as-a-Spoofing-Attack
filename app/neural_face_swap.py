"""
Neural Face Swap — Integração Oficial FaceFusion
================================================================
Utiliza o motor interno do FaceFusion para alinhar, processar e mesclar a face,
garantindo a melhor qualidade e evitando distorções.
"""

import os
import sys
import cv2
import numpy as np

# Injeta facefusion no path
FF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "facefusion")
if FF_DIR not in sys.path:
    sys.path.insert(0, FF_DIR)

from facefusion import state_manager
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.processors.modules.face_swapper.core import swap_face


class NeuralFaceSwapper:
    """Face swap neural utilizando a API interna do FaceFusion."""

    def __init__(self, source_image: np.ndarray, source_bbox: tuple = None):
        """
        Inicializa o estado global do FaceFusion.
        """
        print("🧠 Inicializando motor oficial do FaceFusion...")
        
        # Configuração do estado global do FaceFusion
        config = {
            'execution_providers': ['tensorrt', 'cuda', 'cpu'],
            'face_swapper_model': 'inswapper_128',
            'face_detector_model': 'yolo_face',
            'face_detector_size': '640x640',
            'face_detector_score': 0.5,
            'face_landmarker_score': 0.5,
            'face_mask_types': ['box'],
            'face_mask_blur': 0.3,
            'face_mask_padding': (0, 0, 0, 0),
            'face_swapper_weight': 1.0,
            'face_swapper_pixel_boost': '128x128',
            'download_providers': ['github'],
            'face_detector_angles': [0],
            'face_detector_margin': (0, 0, 0, 0),
            'execution_device_ids': ['0', '1'],
            'execution_thread_count': 8,
            'face_landmarker_model': '2dfan4',
            'face_recognizer_model': 'arcface_w600k_r50'
        }
        
        for k, v in config.items():
            state_manager.init_item(k, v)

        # Analisa a face do atacante (fonte)
        faces = get_many_faces([source_image])
        self.source_face = get_one_face(faces)
        
        if not self.source_face:
            print("⚠️ FaceFusion não encontrou a face na imagem original. Tentando fallback via resize...")
            resized = cv2.resize(source_image, (640,640))
            faces_fallback = get_many_faces([resized])
            self.source_face = get_one_face(faces_fallback)
            
        print(f"   ✅ Atacante processado via FaceFusion: {'Sim' if self.source_face else 'Não'}")


    def swap(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Aplica o swap neural no frame usando FaceFusion.
        O bbox é ignorado aqui pois o FaceFusion detectará a face internamente.
        """
        if self.source_face is None:
            return frame
            
        # O FaceFusion detecta a face e seus landmarks no frame alvo
        target_faces = get_many_faces([frame])
        if not target_faces:
            return frame
            
        # Pega a face alvo (a maior ou primeira detectada)
        target_face = target_faces[0]
        
        # Opcional: Se quisermos usar o bbox exato do servidor para escolher a face
        # poderíamos filtrar `target_faces` com base no IoU com `bbox`.
        # Mas para a maioria dos casos simples (1 pessoa), target_faces[0] é perfeito.
        
        # Aplica o swap neural usando a engine completa do FF
        output_frame = swap_face(self.source_face, target_face, frame)
        
        return output_frame
