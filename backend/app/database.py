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

class RostoDB(Base):
    __tablename__ = "rostos"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String(100), nullable=False)
    embedding = Column(Vector(128)) 

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()