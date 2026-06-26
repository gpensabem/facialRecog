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

from app.database import RostoDB, UsuarioDB

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
    nome: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    extractor: FaceExtractorInterface = Depends(get_face_extractor)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    # 1. Recorta apenas o rosto da imagem enviada
    try:
        imagem_rosto = detector.get_face_crop(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Extrai o vetor de características
    embedding = extractor.extract_embedding(imagem_rosto)
    
    # Garante que o vetor seja uma lista de floats pura antes de mandar pro Postgres
    if isinstance(embedding, np.ndarray):
        embedding = embedding.flatten().tolist()

    # 3. FLUXO INTELIGENTE 1:N (Evita IDs duplicados para a mesma pessoa)
    # Procura se já existe um usuário com esse mesmo nome exato no banco
    usuario = db.query(UsuarioDB).filter(UsuarioDB.nome == nome).first()
    
    if not usuario:
        # Se o usuário não existe, criamos a "Classe" dele na tabela de usuários
        usuario = UsuarioDB(nome=nome)
        db.add(usuario)
        db.commit()
        db.refresh(usuario) # Pega o ID gerado automaticamente pelo banco
        status_msg = "Novo usuário criado e primeira face cadastrada!"
    else:
        status_msg = f"Nova variação de rosto adicionada ao usuário '{nome}' com sucesso!"

    # 4. Vincula a nova foto ao ID do usuário encontrado ou criado
    novo_rosto = RostoDB(usuario_id=usuario.id, embedding=embedding)
    db.add(novo_rosto)
    db.commit()
    
    # Busca o total de fotos que esse usuário já tem para dar um feedback legal no JSON
    total_faces = db.query(RostoDB).filter(RostoDB.usuario_id == usuario.id).count()
    
    return {
        "usuario_id": usuario.id, 
        "nome": usuario.nome, 
        "total_fotos_vinculadas": total_faces,
        "status": status_msg
    }

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

    # 1. Recorta o rosto da imagem ao vivo da câmera
    try:
        imagem_rosto = detector.get_face_crop(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Extrai o vetor do rosto focado
    embedding_atual = extractor.extract_embedding(imagem_rosto)
    
    # Garante que o vetor seja uma lista de floats para o pgvector trabalhar sem problemas
    if isinstance(embedding_atual, np.ndarray):
        embedding_atual = embedding_atual.flatten().tolist()

    # 3. Busca o vetor mais próximo na tabela de rostos
    rosto_mais_proximo = db.query(RostoDB)\
        .order_by(RostoDB.embedding.cosine_distance(embedding_atual))\
        .first()

    if not rosto_mais_proximo:
        raise HTTPException(status_code=404, detail="Nenhum usuário cadastrado.")

    # 4. Calcula a distância cosseno
    distancia = db.query(
        RostoDB.embedding.cosine_distance(embedding_atual)
    ).filter(RostoDB.id == rosto_mais_proximo.id).scalar()

    # 5. Validação com base no threshold configurado
    if distancia <= settings.threshold:
        # A MÁGICA ESTÁ AQUI: Pegamos o nome através do relacionamento .usuario
        usuario_dono = rosto_mais_proximo.usuario
        
        return {
            "reconhecido": True,
            "usuario_id": usuario_dono.id,      # ID único do usuário
            "usuario": usuario_dono.nome,        # Nome vindo da tabela de usuários
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