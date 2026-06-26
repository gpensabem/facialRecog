import os
import urllib.request
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class FaceLivenessDetector:
    def __init__(self, threshold: float = 0.85):
        """
        Inicializa o detector de prova de vida carregando um modelo ONNX real.
        """
        self.threshold = threshold
        
        # Define o caminho absoluto para a pasta /models na raiz do projeto
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.model_dir = os.path.join(base_dir, "models")
        self.model_path = os.path.join(self.model_dir, "minifasnet_v2.onnx")
        
        # Link direto para um modelo ONNX público (MiniFASNet - arquitetura focada em Anti-Spoofing)
        # Nota: Este é um link de um fork aberto focado em ONNX.
        self.model_url = "https://github.com/kprokofi/light-weight-face-anti-spoofing/raw/master/models/anti_spoof_models/MiniFASNetV2.onnx"
        
        self._ensure_model_exists()
        self._load_model()

    def _ensure_model_exists(self):
        """Cria a pasta e baixa o modelo automaticamente se ele não existir."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logger.info(f"📂 Pasta {self.model_dir} criada.")
            
        if not os.path.exists(self.model_path):
            logger.info("⏳ Modelo ONNX de Liveness não encontrado. Baixando automaticamente (isso pode levar alguns segundos)...")
            try:
                # Disfarça o urllib como um navegador para evitar bloqueios de firewall do GitHub
                req = urllib.request.Request(self.model_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(self.model_path, 'wb') as out_file:
                    out_file.write(response.read())
                logger.info("✅ Download do modelo concluído com sucesso!")
            except Exception as e:
                logger.error(f"❌ Erro ao baixar o modelo: {e}")
                raise RuntimeError(f"Falha ao baixar modelo. Baixe manualmente e coloque em: {self.model_path}")

    def _load_model(self):
        """Carrega o modelo ONNX para a memória usando o OpenCV."""
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            logger.info("🧠 Modelo de Liveness carregado na memória via OpenCV.")
        except Exception as e:
            logger.error(f"❌ Erro ao ler o arquivo ONNX. O arquivo pode estar corrompido: {e}")
            raise

    def evaluate(self, face_crop: np.ndarray) -> dict:
        """
        Avalia se a imagem (rosto cortado) é real ou uma tentativa de fraude.
        """
        try:
            # 1. Prepara a imagem para a IA
            # A maioria dos modelos MiniFASNet espera imagens 80x80 pixels.
            # Convertendo para o formato de "Blob" (Tensor) que a rede neural entende
            blob = cv2.dnn.blobFromImage(
                face_crop, 
                scalefactor=1.0, 
                size=(80, 80), 
                mean=(0, 0, 0), 
                swapRB=True, # OpenCV usa BGR, o modelo geralmente espera RGB
                crop=False
            )
            
            # 2. Roda a inferência (Passa a imagem pela rede)
            self.net.setInput(blob)
            out = self.net.forward()
            
            # 3. Interpreta a saída
            # Modelos de Liveness geralmente retornam um array de probabilidades (Softmax).
            # Por padrão no MiniFASNet: índice 0 = Spoof (Fraude), índice 1 = Real.
            # O array costuma vir no formato: [[prob_fake, prob_real, ...]]
            score_real = float(out[0][1])
            
            is_real = score_real >= self.threshold
            
            return {
                "is_real": is_real,
                "score": round(score_real, 3)
            }
            
        except Exception as e:
            logger.error(f"⚠️ Erro durante a inferência de Liveness: {e}")
            # Em caso de pane no processamento, barramos o acesso por segurança
            return {"is_real": False, "score": 0.0}