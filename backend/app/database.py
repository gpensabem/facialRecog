from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from app.config import settings

engine = create_engine(settings.database_url)
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