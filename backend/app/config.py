import os
import yaml

class Settings:
    def __init__(self):
        # Garante que vai achar o config.yml na raiz do projeto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config.yml")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    @property
    def database_url(self) -> str:
        return self._config["database"]["url"]

    @property
    def model_path(self) -> str:
        return self._config["face_recognition"]["model_path"]

    @property
    def threshold(self) -> float:
        return float(self._config["face_recognition"]["threshold"])
    
    @property
    def face_liveness_threshold(self) -> float:
        return float(self._config["face_liveness"]["threshold"])

# Instância global configurada
settings = Settings()