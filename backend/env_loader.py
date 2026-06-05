import os
import logging
from pathlib import Path
from dotenv import load_dotenv

def load_selected_env(project_root: Path = None):
    """
    Procura e carrega as variáveis de ambiente do arquivo .env
    para a memória do sistema (os.environ).
    """
    # Se não for passado um diretório raiz, assume o diretório atual deste arquivo
    if project_root is None:
        project_root = Path(__file__).resolve().parent
        
    env_path = project_root / '.env'
    
    if env_path.exists():
        # Carrega o arquivo .env substituindo variáveis já existentes (override=True)
        load_dotenv(dotenv_path=env_path, override=True)
        logging.info(f"Variáveis de ambiente carregadas com sucesso de: {env_path}")
    else:
        logging.warning(f"Nenhum arquivo .env encontrado em {env_path}. O sistema usará as variáveis nativas do SO.")

# Bloco de teste rápido (opcional): se você rodar `python env_loader.py` direto, ele testa se funcionou.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_selected_env()
    print("Database URL:", os.getenv("DATABASE_URL", "Não configurada"))