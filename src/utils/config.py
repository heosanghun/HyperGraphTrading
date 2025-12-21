"""
설정 관리 모듈
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 환경 변수 로드
# 프로젝트 루트의 상위 디렉토리에서 env/.env 파일 찾기
env_path = Path(__file__).parent.parent.parent.parent / "env" / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # 현재 디렉토리에서 .env 찾기
    load_dotenv()


class Config:
    """설정 관리 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """설정 초기화"""
        self.config: Dict[str, Any] = {}
        
        if config_path:
            self.load_from_file(config_path)
        
        # 환경 변수에서 설정 로드
        self.load_from_env()
    
    def load_from_file(self, config_path: str) -> None:
        """YAML 파일에서 설정 로드"""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                self.config.update(yaml.safe_load(f) or {})
    
    def load_from_env(self) -> None:
        """환경 변수에서 설정 로드"""
        env_config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "database_url": os.getenv("DATABASE_URL"),
            "redis_url": os.getenv("REDIS_URL"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
        
        # None이 아닌 값만 업데이트
        for key, value in env_config.items():
            if value is not None:
                self.config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 조회"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """설정 값 설정"""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """설정 업데이트"""
        self.config.update(updates)

