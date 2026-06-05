import cv2
import numpy as np
import os
import urllib.request

class FaceDetector:
    def __init__(self):
        # 1. Encontra a raiz do projeto (mvp_face_recog)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(base_dir, "models")
        
        # Garante que a pasta models exista
        os.makedirs(models_dir, exist_ok=True)
        
        # Define o caminho seguro dentro do nosso projeto
        cascade_path = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')
        
        # 2. Se o arquivo não existir, baixa direto do repositório oficial do OpenCV
        if not os.path.exists(cascade_path):
            print("XML do Haar Cascade não encontrado. Baixando do repositório oficial do OpenCV...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, cascade_path)
            print("Download concluído!")

        # 3. O SEGREDO ESTÁ AQUI: Transforma o caminho absoluto em relativo
        # Isso remove o "Área de Trabalho" da string e evita o bug do OpenCV
        caminho_relativo = os.path.relpath(cascade_path)
        
        # Carrega o modelo de forma local usando o caminho sem acentos
        self.detector = cv2.CascadeClassifier(caminho_relativo)
        
        if self.detector.empty():
            raise RuntimeError(
                "Falha crítica: O OpenCV continua recusando o caminho. "
                "Para resolver, mova a pasta do projeto (tergo2026) direto para o C:/ "
                "para remover os acentos do caminho."
            )
        
    def get_face_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Encontra o rosto na imagem original, remove o fundo e retorna apenas a face com margem.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        
        if len(faces) == 0:
            raise ValueError("Nenhum rosto humano detectado na imagem. Tente outra foto.")
            
        # Pega o maior rosto detectado
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        
        y1 = max(0, y - pad_y)
        y2 = min(image.shape[0], y + h + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(image.shape[1], x + w + pad_x)
        
        rosto_recortado = image[y1:y2, x1:x2]
        
        return rosto_recortado