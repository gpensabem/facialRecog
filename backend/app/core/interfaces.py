from abc import ABC, abstractmethod
import numpy as np

class FaceExtractorInterface(ABC):
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Carrega o modelo ONNX em memória."""
        pass

    @abstractmethod
    def extract_embedding(self, face_image: np.ndarray) -> list[float]:
        """Recebe a imagem do rosto cortado e retorna o vetor numérico."""
        pass


class FaceDetectorInterface(ABC):
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Carrega o modelo ONNX de detecção em memória."""
        pass

    @abstractmethod
    def detect_and_align(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Recebe a imagem inteira.
        Deve detectar as faces, encontrar os landmarks (olhos, nariz, boca),
        aplicar a transformação afim (alinhamento) e retornar os recortes (112x112).
        """
        pass