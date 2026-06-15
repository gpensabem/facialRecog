import logging
from sqlalchemy import text
# Ajuste os imports abaixo de acordo com a pasta real onde estão seus arquivos
from app.database import engine, Base, RostoDB 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    logger.info("Iniciando configuração do banco de dados...")

    try:
        # Passo 1: O pgvector exige que a extensão seja criada no banco ANTES da tabela
        with engine.connect() as conn:
            logger.info("Ativando a extensão pgvector (se não existir)...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            
        # Passo 2: Lê os seus modelos (RostoDB) e cria as tabelas no PostgreSQL
        logger.info("Criando as tabelas baseadas no SQLAlchemy...")
        Base.metadata.create_all(bind=engine)
        
        logger.info("Banco de dados configurado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro ao configurar o banco de dados: {e}")
        logger.info("DICA: Verifique se o PostgreSQL que está rodando possui o pgvector instalado.")

if __name__ == "__main__":
    init_database()