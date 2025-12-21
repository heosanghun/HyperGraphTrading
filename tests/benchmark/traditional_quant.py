"""
Traditional Quant (XGBoost & LSTM) 베이스라인 모델
논문 5.1.2 비교 모델(Baselines)에 명시된 전통적 퀀트 모델
"""
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# XGBoost 및 LSTM 라이브러리
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost를 설치할 수 없습니다. pip install xgboost")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch를 설치할 수 없습니다. pip install torch")

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.trading.backtester import Backtester


class LSTMPredictor(nn.Module):
    """LSTM 기반 시계열 예측 모델"""
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 마지막 시퀀스만 사용
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output


class TraditionalQuant:
    """Traditional Quant (XGBoost & LSTM) 통합 모델"""
    
    def __init__(self, 
                 xgb_weight: float = 0.6,
                 lstm_weight: float = 0.4,
                 sequence_length: int = 30,
                 device: str = "cpu"):
        """Traditional Quant 모델 초기화"""
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.sequence_length = sequence_length
        self.device = device
        
        # 모델 초기화
        self.xgb_model = None
        self.lstm_model = None
        
        # 데이터 전처리기
        self.preprocessor = DataPreprocessor()
        
        # 모델 학습 여부
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """XGBoost용 특징 준비"""
        df_features = self.preprocessor.engineer_features(df)
        
        # 기술적 지표 추가
        if "close" in df_features.columns:
            close = df_features["close"]
            
            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df_features["macd"] = ema12 - ema26
            df_features["macd_signal"] = df_features["macd"].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            df_features["bb_upper"] = sma20 + (std20 * 2)
            df_features["bb_lower"] = sma20 - (std20 * 2)
            df_features["bb_width"] = (df_features["bb_upper"] - df_features["bb_lower"]) / sma20
            
            # ATR (Average True Range)
            if "high" in df_features.columns and "low" in df_features.columns:
                high_low = df_features["high"] - df_features["low"]
                high_close = np.abs(df_features["high"] - close.shift())
                low_close = np.abs(df_features["low"] - close.shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df_features["atr"] = tr.rolling(window=14).mean()
        
        # 결측치 처리
        df_features = self.preprocessor.handle_missing_values(df_features)
        
        return df_features
    
    def prepare_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """LSTM용 시계열 데이터 준비"""
        # OHLCV 데이터 선택
        ohlcv_cols = []
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                ohlcv_cols.append(col)
            elif col.capitalize() in df.columns:
                ohlcv_cols.append(col.capitalize())
        
        if len(ohlcv_cols) < 4:
            # 최소한 close와 volume만이라도
            if "close" in df.columns or "Close" in df.columns:
                close_col = "close" if "close" in df.columns else "Close"
                df_seq = df[[close_col]].copy()
                if "volume" in df.columns or "Volume" in df.columns:
                    vol_col = "volume" if "volume" in df.columns else "Volume"
                    df_seq[vol_col] = df[vol_col]
                else:
                    df_seq["volume"] = 1.0
            else:
                return np.array([])
        else:
            df_seq = df[ohlcv_cols].copy()
        
        # 정규화
        df_seq = self.preprocessor.normalize_timeseries(df_seq, df_seq.columns.tolist())
        
        # 시퀀스 생성
        sequences = []
        for i in range(self.sequence_length, len(df_seq)):
            seq = df_seq.iloc[i-self.sequence_length:i].values
            sequences.append(seq)
        
        return np.array(sequences) if sequences else np.array([])
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray):
        """XGBoost 모델 학습"""
        if not XGBOOST_AVAILABLE:
            return None
        
        # 방향성 레이블 생성 (상승/하락/보합)
        y_direction = np.zeros(len(y_train))
        y_direction[y_train > 0.01] = 1  # 상승
        y_direction[y_train < -0.01] = -1  # 하락
        # 나머지는 0 (보합)
        
        # XGBoost 모델 학습
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.xgb_model.fit(X_train, y_direction)
        return self.xgb_model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10):
        """LSTM 모델 학습"""
        if not TORCH_AVAILABLE or len(X_train) == 0:
            return None
        
        input_size = X_train.shape[2]
        self.lstm_model = LSTMPredictor(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=1
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        self.lstm_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.lstm_model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
        
        return self.lstm_model
    
    def train(self, price_data: Dict[str, pd.DataFrame], symbols: List[str]):
        """모델 학습"""
        print("[INFO] Traditional Quant 모델 학습 시작...")
        
        all_X_xgb = []
        all_y = []
        all_X_lstm = []
        all_y_lstm = []
        
        for symbol in symbols:
            if symbol not in price_data:
                continue
            
            df = price_data[symbol].copy()
            if len(df) < self.sequence_length + 10:
                continue
            
            # 특징 준비
            df_features = self.prepare_features(df)
            
            # XGBoost용 데이터
            feature_cols = [col for col in df_features.columns 
                          if col not in ["open", "high", "low", "close", "volume", 
                                        "Open", "High", "Low", "Close", "Volume"]]
            
            if len(feature_cols) > 0:
                X_xgb = df_features[feature_cols].values
                # 수익률 계산 (다음 날 대비)
                if "close" in df.columns:
                    returns = df["close"].pct_change().shift(-1).fillna(0).values
                else:
                    returns = df["Close"].pct_change().shift(-1).fillna(0).values
                
                # 유효한 데이터만
                valid_idx = ~(np.isnan(X_xgb).any(axis=1) | np.isnan(returns))
                X_xgb = X_xgb[valid_idx]
                y = returns[valid_idx]
                
                if len(X_xgb) > 0:
                    all_X_xgb.append(X_xgb)
                    all_y.append(y)
            
            # LSTM용 데이터
            sequences = self.prepare_sequences(df)
            if len(sequences) > 0:
                # 다음 날 수익률
                if "close" in df.columns:
                    future_returns = df["close"].pct_change().shift(-1).fillna(0).values
                else:
                    future_returns = df["Close"].pct_change().shift(-1).fillna(0).values
                
                y_lstm = future_returns[self.sequence_length:]
                
                if len(sequences) == len(y_lstm):
                    all_X_lstm.append(sequences)
                    all_y_lstm.append(y_lstm)
        
        # 데이터 결합
        if len(all_X_xgb) > 0:
            X_xgb_combined = np.vstack(all_X_xgb)
            y_combined = np.hstack(all_y)
            
            # XGBoost 학습
            print("[INFO] XGBoost 모델 학습 중...")
            self.train_xgboost(X_xgb_combined, y_combined)
        
        if len(all_X_lstm) > 0:
            X_lstm_combined = np.vstack(all_X_lstm)
            y_lstm_combined = np.hstack(all_y_lstm)
            
            # LSTM 학습
            print("[INFO] LSTM 모델 학습 중...")
            self.train_lstm(X_lstm_combined, y_lstm_combined, epochs=5)
        
        self.is_trained = True
        print("[INFO] Traditional Quant 모델 학습 완료")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """예측 수행"""
        if not self.is_trained:
            return {"decision": "HOLD", "confidence": 0.5, "xgb_pred": 0, "lstm_pred": 0.0}
        
        # XGBoost 예측
        xgb_pred = 0
        xgb_confidence = 0.5
        
        if self.xgb_model is not None and XGBOOST_AVAILABLE:
            try:
                df_features = self.prepare_features(df)
                feature_cols = [col for col in df_features.columns 
                              if col not in ["open", "high", "low", "close", "volume",
                                            "Open", "High", "Low", "Close", "Volume"]]
                
                if len(feature_cols) > 0:
                    X_xgb = df_features[feature_cols].iloc[-1:].values
                    if not np.isnan(X_xgb).any():
                        xgb_pred = self.xgb_model.predict(X_xgb)[0]
                        xgb_proba = self.xgb_model.predict_proba(X_xgb)[0]
                        xgb_confidence = np.max(xgb_proba)
            except Exception as e:
                print(f"[WARNING] XGBoost 예측 오류: {e}")
        
        # LSTM 예측
        lstm_pred = 0.0
        lstm_confidence = 0.5
        
        if self.lstm_model is not None and TORCH_AVAILABLE:
            try:
                sequences = self.prepare_sequences(df)
                if len(sequences) > 0:
                    X_lstm = sequences[-1:]
                    X_tensor = torch.FloatTensor(X_lstm).to(self.device)
                    
                    self.lstm_model.eval()
                    with torch.no_grad():
                        lstm_output = self.lstm_model(X_tensor)
                        lstm_pred = lstm_output.cpu().numpy()[0, 0]
                        lstm_confidence = min(abs(lstm_pred) * 10, 1.0)  # 수익률 크기 기반 신뢰도
            except Exception as e:
                print(f"[WARNING] LSTM 예측 오류: {e}")
        
        # 앙상블 결정
        # XGBoost: 방향성 (-1, 0, 1)
        # LSTM: 수익률 예측값
        
        # 가중 평균
        if xgb_pred > 0 and lstm_pred > 0:
            decision = "BUY"
            confidence = (xgb_confidence * self.xgb_weight + lstm_confidence * self.lstm_weight)
        elif xgb_pred < 0 and lstm_pred < 0:
            decision = "SELL"
            confidence = (xgb_confidence * self.xgb_weight + lstm_confidence * self.lstm_weight)
        elif abs(lstm_pred) > 0.02:  # LSTM이 강한 신호
            decision = "BUY" if lstm_pred > 0 else "SELL"
            confidence = lstm_confidence * 0.8
        else:
            decision = "HOLD"
            confidence = 0.5
        
        return {
            "decision": decision,
            "confidence": min(confidence, 1.0),
            "xgb_pred": int(xgb_pred),
            "lstm_pred": float(lstm_pred),
            "xgb_confidence": float(xgb_confidence),
            "lstm_confidence": float(lstm_confidence)
        }
    
    def test(self, 
             symbols: List[str],
             start_date: str,
             end_date: str) -> Dict[str, Any]:
        """테스트 실행"""
        print("\n" + "="*80)
        print("Traditional Quant (XGBoost & LSTM) 테스트")
        print("="*80)
        
        start_time = time.time()
        
        # 데이터 수집
        print("[1/4] 데이터 수집 중...")
        collector = DataCollector()
        price_data = collector.collect_price_data(symbols, start_date, end_date)
        
        if not price_data:
            return self._get_default_result()
        
        # 모델 학습
        print("[2/4] 모델 학습 중...")
        self.train(price_data, symbols)
        
        # 백테스팅
        print("[3/4] 백테스팅 실행 중...")
        backtester = Backtester(initial_capital=10000.0)
        inference_times = []
        
        for symbol in symbols[:1]:  # 첫 번째 심볼만 테스트
            if symbol not in price_data:
                continue
            
            df = price_data[symbol].copy()
            decisions = []
            
            for i in range(self.sequence_length + 10, len(df)):
                # 현재까지의 데이터로 예측
                df_current = df.iloc[:i+1]
                
                # 추론 시간 측정
                infer_start = time.time()
                prediction = self.predict(df_current)
                infer_time = (time.time() - infer_start) * 1000  # ms
                inference_times.append(infer_time)
                
                decisions.append({
                    "decision": prediction["decision"],
                    "confidence": prediction["confidence"],
                    "timestamp": df.index[i] if hasattr(df.index[i], 'strftime') else str(i)
                })
            
            # 백테스팅 실행
            backtest_result = backtester.run_backtest(df, decisions)
            metrics = backtest_result["metrics"]
        
        total_time = time.time() - start_time
        
        result = {
            "total_time_seconds": total_time,
            "avg_inference_time_ms": np.mean(inference_times) if inference_times else 0.1,
            "p95_inference_time_ms": np.percentile(inference_times, 95) if inference_times else 0.2,
            "p99_inference_time_ms": np.percentile(inference_times, 99) if inference_times else 0.3,
            "total_return": metrics.get("total_return", 0.0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "total_trades": metrics.get("total_trades", 0),
            "api_calls": 0,
            "cost_usd": 0.0,
            "model_type": "Traditional Quant (XGBoost & LSTM)",
            "xgb_available": XGBOOST_AVAILABLE,
            "lstm_available": TORCH_AVAILABLE
        }
        
        print(f"\n[완료] Traditional Quant 테스트 완료")
        print(f"   총 수익률: {result['total_return']*100:.2f}%")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"   최대 낙폭: {result['max_drawdown']*100:.2f}%")
        print(f"   추론 시간: {result['avg_inference_time_ms']:.3f}ms")
        
        return result
    
    def _get_default_result(self) -> Dict[str, Any]:
        """기본 결과 반환"""
        return {
            "total_time_seconds": 0.0,
            "avg_inference_time_ms": 0.1,
            "p95_inference_time_ms": 0.2,
            "p99_inference_time_ms": 0.3,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "api_calls": 0,
            "cost_usd": 0.0,
            "model_type": "Traditional Quant (XGBoost & LSTM)",
            "xgb_available": XGBOOST_AVAILABLE,
            "lstm_available": TORCH_AVAILABLE
        }


if __name__ == "__main__":
    """단독 테스트"""
    print("Traditional Quant (XGBoost & LSTM) 단독 테스트")
    
    model = TraditionalQuant()
    result = model.test(
        symbols=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print(f"\n결과: {result}")

