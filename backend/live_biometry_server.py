"""
Servidor WebSocket para Reconhecimento Facial ao Vivo.
Execute separado do worker e da API principal:
    uvicorn live_biometry_server:live_face_app --host 0.0.0.0 --port 8004
"""
import os
import cv2
import json
import time
import logging
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
from sqlalchemy import text # Necessário para rodar queries diretas do pgvector

from app.database import RostoDB, UsuarioDB
from app.config import settings


# Carrega as variáveis de ambiente
from env_loader import load_selected_env
load_selected_env(project_root=Path(__file__).resolve().parent)

# --- IMPORTS AJUSTADOS PARA A ESTRUTURA MVP ---
# Substitua 'get_db' pelo nome real da função/variável no seu app/database.py
from app.database import SessionLocal 

# Importa as classes diretamente de app/services
# Lembre-se de verificar se a classe dentro do arquivo se chama realmente 'MobileFaceNet' e 'FaceDetector'
from app.services.mobilefacenet import MobileFaceNetExtractor
from app.services.detector import FaceDetector
from app.services.liveness_detector import FaceLivenessDetector

MODEL_FOLDER = os.getenv("MODEL_FOLDER", "models")
# Alterado para o nome exato do arquivo que aparece no seu print
FACE_MODEL_NAME = "mobilefacenet.onnx" 

# Instâncias globais
face_extractor = None
face_detector = FaceDetector()

face_liveness = FaceLivenessDetector(threshold=settings.face_liveness_threshold) # <--- NOVO

@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_extractor
    logging.info("Iniciando validação e carregamento do modelo de Biometria...")
    
    loop = asyncio.get_event_loop()
    model_path = os.path.join(MODEL_FOLDER, FACE_MODEL_NAME)
    
    # Inicializa o seu extrator ONNX do MVP
    # Verifique se a sua classe MobileFaceNet recebe o 'model_path' no __init__
    face_extractor = await loop.run_in_executor(
        None, lambda: MobileFaceNetExtractor(model_path=model_path)
    )
    
    logging.info("Modelo de reconhecimento facial carregado com sucesso!")
    yield
    
    global _infer_executor, _db_executor
    _infer_executor.shutdown(wait=True)
    _db_executor.shutdown(wait=True)

live_face_app = FastAPI(title="IAVision Face Live", lifespan=lifespan)

_infer_executor = ThreadPoolExecutor(max_workers=1)
_db_executor = ThreadPoolExecutor(max_workers=2)

def _process_face_frame(frame: np.ndarray, detector, liveness, extractor) -> tuple[list[float] | None, dict]:
    """
    1. Corta o rosto
    2. Valida Liveness (Prova de Vida)
    3. Extrai o embedding (apenas se for real)
    """
    try:
        imagem_rosto = detector.get_face_crop(frame)
    except ValueError:
        return None, {"is_real": False, "score": 0.0}

    # Avalia se a imagem é de uma pessoa viva ou de uma foto/tela
    liveness_result = liveness.evaluate(imagem_rosto)
    
    # Se for fraude/spoofing, abortamos aqui. Não extrai vetor.
    if not liveness_result["is_real"]:
        return None, liveness_result

    # Só extrai o vetor se a pessoa for real
    embedding = extractor.extract_embedding(imagem_rosto)
    
    if embedding is not None:
        if isinstance(embedding, np.ndarray):
            return embedding.flatten().tolist(), liveness_result
        return embedding, liveness_result
        
    return None, liveness_result


def _query_closest_face(embedding: list[float]) -> dict | None:
    db = SessionLocal()
    try:
        # Busca o vetor mais próximo na tabela de rostos
        rosto_mais_proximo = db.query(RostoDB)\
            .order_by(RostoDB.embedding.cosine_distance(embedding))\
            .first()

        if not rosto_mais_proximo:
            return None

        # Calcula a distância desse vetor
        distancia = db.query(
            RostoDB.embedding.cosine_distance(embedding)
        ).filter(RostoDB.id == rosto_mais_proximo.id).scalar()

        if distancia <= settings.threshold:
            # A GRANDE MUDANÇA: Retornamos os dados do USUÁRIO dono daquela face
            usuario_dono = rosto_mais_proximo.usuario
            
            return {
                "id": str(usuario_dono.id), # ID único da pessoa
                "nome": usuario_dono.nome,  # Nome da pessoa
                "confidence": round(1.0 - float(distancia), 4)
            }
        
        return None
    finally:
        db.close()

@live_face_app.websocket("/ws/face-recognition")
async def face_recognition_websocket(websocket: WebSocket):
    await websocket.accept()
    logging.info("Novo cliente conectado para Reconhecimento Facial.")
    
    loop = asyncio.get_event_loop()
    busy = False

    try:
        while True:
            raw = await websocket.receive_bytes()

            if busy:
                continue  # Ignora se estiver processando o frame anterior

            busy = True
            
            # Decodifica a imagem com segurança
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                busy = False
                continue

            t0 = time.perf_counter()
            
            # Inicializa as variáveis padrão para este frame
            face_detected = False
            resultado = None

            try:
                # Extrai o embedding E o status de liveness simultaneamente
                embedding, liveness_status = await loop.run_in_executor(
                    _infer_executor, _process_face_frame, frame, face_detector, face_liveness, face_extractor
                )
                
                
                # Se detectou um rosto e passou no teste de vida, vai pro banco!
                if embedding is not None:
                    face_detected = True
                    resultado = await loop.run_in_executor(
                        _db_executor, _query_closest_face, embedding
                    )
                # Se detectou rosto, mas é uma foto/fraude (Liveness falhou)
                elif liveness_status["score"] > 0.0 and not liveness_status["is_real"]:
                    face_detected = True
                    resultado = None # Barramos a busca no banco

            except Exception as inner_exc:
                logging.error(f"Erro ao processar frame/banco: {inner_exc}")
                liveness_status = {"is_real": False, "score": 0.0}
                resultado = None

            inference_ms = int((time.perf_counter() - t0) * 1000)
            busy = False

            # O JSON agora carrega o status antifraude para o Frontend brilhar!
            await websocket.send_text(json.dumps({
                "face_detected": face_detected,
                "liveness": liveness_status, # Ex: {"is_real": True, "score": 0.98}
                "match": resultado,
                "inference_ms": inference_ms
            }))

            inference_ms = int((time.perf_counter() - t0) * 1000)
            busy = False

            # Envia a resposta SEMPRE, mesmo que dê "Desconhecido" ou não ache rosto
            # O Frontend precisa receber isso para continuar enviando os próximos frames!
            await websocket.send_text(json.dumps({
                "face_detected": face_detected,
                "match": resultado,  # Pode ser um dicionário ou None (Desconhecido)
                "inference_ms": inference_ms
            }))

    except WebSocketDisconnect:
        logging.info("Cliente de Reconhecimento Facial desconectado voluntariamente.")
    except Exception:
        logging.exception("Erro crítico inevitável no servidor de biometria ao vivo")
        try:
            await websocket.close()
        except Exception:
            pass