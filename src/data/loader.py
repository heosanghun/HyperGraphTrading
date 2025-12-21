"""
데이터 로더 모듈
"""
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import pickle


class DataLoader:
    """데이터 로더"""
    
    def __init__(self, data_dir: str = "data/processed"):
        """로더 초기화"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load_price_data(self, symbol: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """주가 데이터 로드"""
        cache_key = f"price_{symbol}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 파일에서 로드
        file_path = self.data_dir / "prices" / f"{symbol}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["timestamp"] = pd.to_datetime(df.get("date", df.get("Date", "")))
            
            if use_cache:
                self._cache[cache_key] = df
            
            return df
        
        return None
    
    def load_batch(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """배치 로드"""
        data = {}
        for symbol in symbols:
            df = self.load_price_data(symbol)
            if df is not None:
                data[symbol] = df
        return data
    
    def clear_cache(self) -> None:
        """캐시 클리어"""
        self._cache.clear()

