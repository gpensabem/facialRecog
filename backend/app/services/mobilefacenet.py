import onnxruntime as ort
import cv2
import numpy as np
from app.core.interfaces import FaceExtractorInterface

import onnxruntime as ort
import cv2
import numpy as np
from app.core.interfaces import FaceExtractorInterface

class MobileFaceNetExtractor(FaceExtractorInterface):
    def __init__(self, model_path: str):
        self.load_model(model_path)
        
    def load_model(self, model_path: str) -> None:
        # Carrega o modelo priorizando CPU
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def extract_embedding(self, face_image: np.ndarray) -> list[float]:
        # Redimensiona para o padrão da MobileFaceNet (112x112)
        resized = cv2.resize(face_image, (112, 112))
        
        # Normalização dos pixels para o intervalo [-1, 1]
        blob = resized.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        
        # Transpõe de (H, W, C) para (C, H, W) e adiciona dimensão de batch (1, C, H, W)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # Roda a inferência real no arquivo ONNX
        outputs = self.session.run(None, {self.input_name: blob})
        embedding = outputs[0][0]
        
        # Normalização L2 para que a distância de cosseno funcione perfeitamente
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding.tolist()
            
        return (embedding / norm).tolist()

# class MobileFaceNetExtractor(FaceExtractorInterface):
#     def __init__(self, model_path: str):
#         self.load_model(model_path)
        
#     def load_model(self, model_path: str) -> None:
#         # Inicializa a sessão ONNX focada em CPU
#         self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
#         self.input_name = self.session.get_inputs()[0].name

#     def extract_embedding(self, face_image: np.ndarray) -> list[float]:
#         # 1. Redimensiona para o tamanho exigido pelo MobileFaceNet (112x112)
#         resized = cv2.resize(face_image, (112, 112))
        
#         # 2. Pré-processamento: Normalização dos pixels (-1 a 1)
#         blob = resized.astype(np.float32)
#         blob = (blob - 127.5) / 128.0
        
#         # 3. Transposição de canais: OpenCV usa (H, W, C) -> ONNX espera (C, H, W)
#         blob = np.transpose(blob, (2, 0, 1))
#         blob = np.expand_dims(blob, axis=0) # Adiciona a dimensão de Batch (1, 3, 112, 112)

#         # 4. Inferência Real
#         outputs = self.session.run(None, {self.input_name: blob})
#         embedding = outputs[0][0]
        
#         # 5. L2 Normalization (Essencial para a Distância de Cosseno do pgvector funcionar)
#         norm = np.linalg.norm(embedding)
#         if norm == 0:
#             return embedding.tolist()
        
#         embedding_normalized = (embedding / norm).tolist()
#         return embedding_normalized