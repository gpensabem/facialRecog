from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
# Provavelmente a sua linha está parecida com isto, basta adicionar o Form
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import cv2
import numpy as np

from app.config import settings
from app.database import get_db, RostoDB
from app.services.factory import get_face_extractor
from app.core.interfaces import FaceExtractorInterface

# IMPORT NOVO: O nosso detector inteligente
from app.services.detector import FaceDetector

app = FastAPI(title="MVP Face Recognition Modular")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Permite o seu front-end Vite
    allow_credentials=True,
    allow_methods=["*"], # Permite GET, POST, DELETE, etc.
    allow_headers=["*"],
)

# Instancia o detector apenas uma vez ao iniciar a API
detector = FaceDetector()

@app.post("/cadastrar", status_code=status.HTTP_201_CREATED)
async def cadastrar_usuario(
    nome: str = Form(...),          # <--- O SEGREDO ESTÁ AQUI
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    extractor: FaceExtractorInterface = Depends(get_face_extractor)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    # 1. NOVO PASSO: Recorta apenas o rosto da imagem enviada
    try:
        imagem_rosto = detector.get_face_crop(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Extrai as características EXATAMENTE do rosto limpo
    embedding = extractor.extract_embedding(imagem_rosto)

    novo_registro = RostoDB(nome=nome, embedding=embedding)
    db.add(novo_registro)
    db.commit()
    
    return {"id": novo_registro.id, "nome": novo_registro.nome, "status": "Rosto isolado e cadastrado!"}


@app.post("/reconhecer")
async def reconhecer_usuario(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    extractor: FaceExtractorInterface = Depends(get_face_extractor)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    # 1. NOVO PASSO: Recorta o rosto da imagem ao vivo da câmera
    try:
        imagem_rosto = detector.get_face_crop(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Extrai o vetor do rosto focado
    embedding_atual = extractor.extract_embedding(imagem_rosto)

    rosto_mais_proximo = db.query(RostoDB)\
        .order_by(RostoDB.embedding.cosine_distance(embedding_atual))\
        .first()

    if not rosto_mais_proximo:
        raise HTTPException(status_code=404, detail="Nenhum usuário cadastrado.")

    distancia = db.query(
        RostoDB.embedding.cosine_distance(embedding_atual)
    ).filter(RostoDB.id == rosto_mais_proximo.id).scalar()

    if distancia <= settings.threshold:
        return {
            "reconhecido": True,
            "usuario": rosto_mais_proximo.nome,
            "distancia_cosseno": round(float(distancia), 4),
            "mensagem": "Acesso liberado!"
        }
    
    return {
        "reconhecido": False,
        "distancia_cosseno": round(float(distancia), 4),
        "mensagem": "Não reconhecido."
    }


@app.delete("/deletar/{rosto_id}", status_code=status.HTTP_200_OK)
def deletar_usuario(rosto_id: int, db: Session = Depends(get_db)):
    """
    Busca um usuário específico pelo ID e o remove do banco de dados.
    """
    # Busca o registro correspondente
    rosto = db.query(RostoDB).filter(RostoDB.id == rosto_id).first()
    
    # Se não existir, retorna 404 (Not Found)
    if not rosto:
        raise HTTPException(
            status_code=404, 
            detail=f"Usuário com ID {rosto_id} não foi encontrado no banco."
        )
    
    # Remove e salva a alteração
    db.delete(rosto)
    db.commit()
    
    return {
        "status": "sucesso", 
        "mensagem": f"Usuário '{rosto.nome}' (ID {rosto_id}) foi removido com sucesso."
    }


@app.delete("/deletar-todos", status_code=status.HTTP_200_OK)
def deletar_todos_usuarios(db: Session = Depends(get_db)):
    """
    Zera a tabela de rostos, apagando absolutamente todos os registros.
    """
    try:
        # Executa um delete em lote (Bulk Delete) que é super performático
        total_removido = db.query(RostoDB).delete()
        db.commit()
        
        return {
            "status": "sucesso", 
            "mensagem": f"O banco de dados foi limpo. Total de {total_removido} registros apagados."
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno ao tentar limpar o banco: {str(e)}"
        )