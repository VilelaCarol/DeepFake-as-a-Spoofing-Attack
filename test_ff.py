import sys, os, cv2

FF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facefusion")
sys.path.insert(0, FF_DIR)

from facefusion import state_manager
config = {
    'execution_providers': ['cuda', 'cpu'],
    'face_swapper_model': 'inswapper_128',
    'face_detector_model': 'yolo_face',
    'face_detector_size': '640x640',
    'face_detector_score': 0.5,
    'face_landmarker_score': 0.5,
    'face_mask_types': ['box'],
    'face_mask_blur': 0.3,
    'face_mask_padding': (0, 0, 0, 0),
    'face_mask_regions': ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'eye-glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip'],
    'face_swapper_weight': 1.0,
    'face_swapper_pixel_boost': '128x128',
    'download_providers': ['github'],
    'face_detector_angles': [0],
    'face_detector_margin': (0, 0, 0, 0),
    'execution_device_ids': ['0'],
    'face_landmarker_model': '2dfan4',
    'face_recognizer_model': 'arcface_w600k_r50'
}
for k, v in config.items():
    state_manager.init_item(k, v)

from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.processors.modules.face_swapper.core import swap_face

img = cv2.imread('data/dataset_pessoal/originais/carolserioclaro.jpg')
faces = get_many_faces([img])
face = get_one_face(faces)
print("Face detectada:", face is not None)

if face is not None:
    res = swap_face(face, face, img)
    print("Swap output shape:", res.shape)
