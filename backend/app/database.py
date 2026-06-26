import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

# 1. Importa o env_loader para garantir que o .env seja lido
from env_loader import load_selected_env

try:
    load_selected_env()
    print('env carregada')
except:
    print('não foi possivel carregar env')

# 2. Busca a variável diretamente do ambiente do sistema
DATABASE_URL = os.getenv("DATABASE_URL")

# Caso a variável não exista com esse nome exato no seu .env, 
# você pode colocar um valor padrão ou levantar um erro.
if not DATABASE_URL:
    raise ValueError("A variável DATABASE_URL não foi encontrada no arquivo .env")

# 3. Cria o engine usando a string obtida
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.database import Base # Ajuste para o seu import real

class UsuarioDB(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String(255), nullable=False)
    # Aqui pode entrar CPF, email, etc.

    # Relacionamento: Um usuário tem várias faces
    faces = relationship("RostoDB", back_populates="usuario", cascade="all, delete-orphan")

class RostoDB(Base):
    __tablename__ = "rostos"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    
    # O tamanho (ex: 128) depende do seu extrator
    embedding = Column(Vector(128)) 

    # Relacionamento reverso
    usuario = relationship("UsuarioDB", back_populates="faces")

# class RostoDB(Base):
#     __tablename__ = "rostos"

#     id = Column(Integer, primary_key=True, index=True)
#     nome = Column(String(100), nullable=False)
#     embedding = Column(Vector(128)) 

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()