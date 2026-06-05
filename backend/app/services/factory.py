
import yaml

from app.config import settings # Seu leitor de YAML

from app.services.mobilefacenet import MobileFaceNetExtractor
from app.core.interfaces import FaceExtractorInterface

from app.services.detector_onnx import RetinaFaceDetector
from app.core.interfaces import FaceDetectorInterface

# Carrega o YAML
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

from app.config import settings
from app.services.mobilefacenet import MobileFaceNetExtractor
from app.core.interfaces import FaceExtractorInterface

def get_face_extractor() -> FaceExtractorInterface:
    # Acessa diretamente a propriedade model_path que criamos no config.py
    caminho_modelo = settings.model_path
    
    return MobileFaceNetExtractor(caminho_modelo)

def get_face_detector() -> FaceDetectorInterface:
    model_type = settings.face_detection.model_type
    path = settings.face_detection.model_path

    if model_type == "retinaface":
        return RetinaFaceDetector(path)
    # elif model_type == "yolov8face":
    #     return YoloV8FaceDetector(path)
    else:
        raise ValueError(f"Detector {model_type} não suportado.")