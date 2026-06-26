-- Ativa a extensão de vetores (obrigatório para o pgvector funcionar)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Cria a tabela de usuários PRIMEIRO
CREATE TABLE IF NOT EXISTS usuarios (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(255) NOT NULL
);

-- 2. Cria a tabela de rostos referenciando o usuário
CREATE TABLE IF NOT EXISTS rostos (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER NOT NULL,
    embedding VECTOR(128),
    CONSTRAINT fk_usuario
        FOREIGN KEY (usuario_id) 
        REFERENCES usuarios(id)
        ON DELETE CASCADE -- Se deletar o usuário, apaga os rostos dele automaticamente
);

-- Opcional, mas recomendado: Cria um índice para acelerar a busca
CREATE INDEX ON rostos USING hnsw (embedding vector_cosine_ops);