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

MODEL_FOLDER = os.getenv("MODEL_FOLDER", "models")
# Alterado para o nome exato do arquivo que aparece no seu print
FACE_MODEL_NAME = "mobilefacenet.onnx" 

# Instâncias globais
face_extractor = None
face_detector = FaceDetector()

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

def _process_face_frame(frame: np.ndarray, detector, extractor) -> list[float] | None:
    """
    Executa a detecção do rosto e a extração do embedding.
    """
    try:
        # 1. O PASSO QUE FALTAVA: Corta o rosto da imagem bruta
        imagem_rosto = detector.get_face_crop(frame)
    except ValueError:
        # Se não detectar ninguém na câmera neste frame, aborta silenciosamente
        return None

    # 2. Extrai o vetor apenas do rosto focado
    embedding = extractor.extract_embedding(imagem_rosto)
    
    if embedding is not None:
        if isinstance(embedding, np.ndarray):
            return embedding.flatten().tolist()
        return embedding
    return None

def _query_closest_face(embedding: list[float]) -> dict | None:
    """
    Executa a busca vetorial (pgvector) no banco de dados Postgres.
    """
    db = SessionLocal()
    try:
        # Trocamos :emb::vector por CAST(:emb AS vector) para evitar o conflito de sintaxe
        sql = text("""
            SELECT id, nome, 1 - (embedding <=> CAST(:emb AS vector)) AS confidence
            FROM rostos
            WHERE 1 - (embedding <=> CAST(:emb AS vector)) > 0.6
            ORDER BY confidence DESC
            LIMIT 1
        """)
        
        # Converte a lista do Python para uma string no formato do PostgreSQL ex: '[0.1, 0.2, ...]'
        emb_str = str(embedding)
        
        result = db.execute(sql, {"emb": emb_str}).fetchone()
        
        if result:
            return {
                "id": str(result.id),
                "nome": result.nome,
                "confidence": round(result.confidence, 3)
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
                # 1. Extrai o embedding do rosto
                embedding = await loop.run_in_executor(
                    _infer_executor, _process_face_frame, frame, face_detector, face_extractor
                )
                if embedding is not None:
                    face_detected = True
                    # 2. Busca o dono do rosto no banco
                    resultado = await loop.run_in_executor(
                        _db_executor, _query_closest_face, embedding
                    )
            except Exception as inner_exc:
                # Se der erro na IA ou no Banco, logamos mas NÃO fechamos o WebSocket
                logging.error(f"Erro ao processar frame/banco: {inner_exc}")
                resultado = None

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