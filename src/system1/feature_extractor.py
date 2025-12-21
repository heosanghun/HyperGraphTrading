"""
System 1 특징 추출기
논문 4.2.1: 틱 데이터 및 오더북 처리
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class OrderbookProcessor:
    """오더북 처리 클래스 (논문 4.2.1)"""
    
    def __init__(self, num_levels: int = 5):
        """오더북 프로세서 초기화"""
        self.num_levels = num_levels
    
    def extract_orderbook_features(self, orderbook_data: Dict[str, Any]) -> np.ndarray:
        """오더북 특징 추출"""
        features = []
        
        # Bid/Ask Price (1~5호가)
        bid_prices = []
        ask_prices = []
        bid_sizes = []
        ask_sizes = []
        
        for i in range(self.num_levels):
            bid_key = f"bid{i+1}_price" if i > 0 else "bid_price"
            ask_key = f"ask{i+1}_price" if i > 0 else "ask_price"
            bid_size_key = f"bid{i+1}_size" if i > 0 else "bid_size"
            ask_size_key = f"ask{i+1}_size" if i > 0 else "ask_size"
            
            # 다양한 키 이름 지원
            bid_price = orderbook_data.get(bid_key) or \
                       orderbook_data.get(f"bid_{i+1}_price") or \
                       orderbook_data.get(f"bids_{i}_price", 0.0)
            ask_price = orderbook_data.get(ask_key) or \
                       orderbook_data.get(f"ask_{i+1}_price") or \
                       orderbook_data.get(f"asks_{i}_price", 0.0)
            bid_size = orderbook_data.get(bid_size_key) or \
                      orderbook_data.get(f"bid_{i+1}_size") or \
                      orderbook_data.get(f"bids_{i}_size", 0.0)
            ask_size = orderbook_data.get(ask_size_key) or \
                      orderbook_data.get(f"ask_{i+1}_size") or \
                      orderbook_data.get(f"asks_{i}_size", 0.0)
            
            bid_prices.append(bid_price)
            ask_prices.append(ask_price)
            bid_sizes.append(bid_size)
            ask_sizes.append(ask_size)
        
        # Spread 계산
        if bid_prices[0] > 0 and ask_prices[0] > 0:
            spread = ask_prices[0] - bid_prices[0]
            spread_pct = spread / ((bid_prices[0] + ask_prices[0]) / 2) if (bid_prices[0] + ask_prices[0]) > 0 else 0.0
        else:
            spread = 0.0
            spread_pct = 0.0
        
        features.append(spread)
        features.append(spread_pct)
        
        # Depth 계산 (총 잔량)
        total_bid_depth = sum(bid_sizes)
        total_ask_depth = sum(ask_sizes)
        features.append(total_bid_depth)
        features.append(total_ask_depth)
        
        # Imbalance 계산 (매수/매도 잔량 불균형 비율)
        if total_bid_depth + total_ask_depth > 0:
            imbalance = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth)
        else:
            imbalance = 0.0
        features.append(imbalance)
        
        # 가중 평균 가격 (Weighted Mid Price)
        if total_bid_depth > 0 and total_ask_depth > 0:
            weighted_bid = sum(bid_prices[i] * bid_sizes[i] for i in range(self.num_levels)) / total_bid_depth
            weighted_ask = sum(ask_prices[i] * ask_sizes[i] for i in range(self.num_levels)) / total_ask_depth
            weighted_mid = (weighted_bid + weighted_ask) / 2
        else:
            weighted_mid = (bid_prices[0] + ask_prices[0]) / 2 if bid_prices[0] > 0 and ask_prices[0] > 0 else 0.0
        features.append(weighted_mid)
        
        # 미시적 압력 (Micro Pressure)
        # 상위 호가의 불균형이 더 중요
        weighted_imbalance = 0.0
        total_weight = 0.0
        for i in range(self.num_levels):
            weight = 1.0 / (i + 1)  # 가까운 호가일수록 높은 가중치
            level_imbalance = (bid_sizes[i] - ask_sizes[i]) / (bid_sizes[i] + ask_sizes[i] + 1e-8)
            weighted_imbalance += weight * level_imbalance
            total_weight += weight
        if total_weight > 0:
            weighted_imbalance /= total_weight
        features.append(weighted_imbalance)
        
        return np.array(features, dtype=np.float32)


class TickDataProcessor:
    """틱 데이터 처리 클래스"""
    
    def __init__(self, window_size: int = 20):
        """틱 데이터 프로세서 초기화"""
        self.window_size = window_size
    
    def extract_tick_features(self, tick_data: Dict[str, Any]) -> np.ndarray:
        """틱 데이터 특징 추출"""
        features = []
        
        # OHLC 데이터
        open_price = tick_data.get("open", tick_data.get("Open", 0.0))
        high_price = tick_data.get("high", tick_data.get("High", 0.0))
        low_price = tick_data.get("low", tick_data.get("Low", 0.0))
        close_price = tick_data.get("close", tick_data.get("Close", tick_data.get("price", 0.0)))
        volume = tick_data.get("volume", tick_data.get("Volume", 0.0))
        
        features.extend([open_price, high_price, low_price, close_price, volume])
        
        # VWAP (거래량 가중 평균 가격)
        if "prices" in tick_data and "volumes" in tick_data:
            prices = tick_data["prices"][-self.window_size:]
            volumes = tick_data["volumes"][-self.window_size:]
            if len(prices) > 0 and len(volumes) > 0 and sum(volumes) > 0:
                vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
            else:
                vwap = close_price
        else:
            vwap = close_price
        features.append(vwap)
        
        # 체결 강도 (Trade Intensity)
        if "prices" in tick_data and len(tick_data["prices"]) >= 2:
            price_changes = np.diff(tick_data["prices"][-self.window_size:])
            trade_intensity = np.sum(np.abs(price_changes)) / len(price_changes) if len(price_changes) > 0 else 0.0
        else:
            trade_intensity = abs(close_price - open_price) if open_price > 0 else 0.0
        features.append(trade_intensity)
        
        return np.array(features, dtype=np.float32)


class TechnicalIndicatorProcessor:
    """기술적 지표 처리 클래스"""
    
    def __init__(self):
        """기술적 지표 프로세서 초기화"""
        pass
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0  # 중립값
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """MACD 계산"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        # EMA 계산
        def ema(data, period):
            alpha = 2.0 / (period + 1)
            ema_values = [data[0]]
            for price in data[1:]:
                ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
            return np.array(ema_values)
        
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        
        if len(ema_fast) < len(ema_slow):
            ema_fast = np.pad(ema_fast, (len(ema_slow) - len(ema_fast), 0), 'constant', constant_values=ema_fast[0])
        
        macd_line = ema_fast[-1] - ema_slow[-1]
        
        # Signal line
        macd_hist = [macd_line]
        if len(prices) >= slow + signal:
            signal_line = ema(np.array([macd_line]), signal)[-1]
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        """Bollinger Bands 계산"""
        if len(prices) < period:
            middle = prices[-1] if len(prices) > 0 else 0.0
            return middle, middle, middle
        
        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        return upper, middle, lower
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """ATR (Average True Range) 계산"""
        if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) >= period:
            atr = np.mean(true_ranges[-period:])
        else:
            atr = np.mean(true_ranges) if true_ranges else 0.0
        
        return atr
    
    def extract_technical_indicators(self, tick_data: Dict[str, Any]) -> np.ndarray:
        """기술적 지표 추출"""
        features = []
        
        # 가격 데이터 추출
        if "prices" in tick_data:
            prices = np.array(tick_data["prices"][-30:])  # 최근 30개
        elif "close" in tick_data:
            prices = np.array([tick_data["close"]])
        else:
            prices = np.array([tick_data.get("price", 0.0)])
        
        if len(prices) == 0:
            prices = np.array([0.0])
        
        # RSI
        rsi = self.calculate_rsi(prices)
        features.append(rsi)
        
        # MACD
        macd, signal, histogram = self.calculate_macd(prices)
        features.extend([macd, signal, histogram])
        
        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(prices)
        features.extend([upper, middle, lower])
        
        # ATR
        if "high" in tick_data and "low" in tick_data and "close" in tick_data:
            high = np.array([tick_data["high"]])
            low = np.array([tick_data["low"]])
            close = np.array([tick_data["close"]])
        else:
            high = prices
            low = prices
            close = prices
        
        atr = self.calculate_atr(high, low, close)
        features.append(atr)
        
        return np.array(features, dtype=np.float32)


class FeatureExtractor:
    """통합 특징 추출기 (논문 4.2.1)"""
    
    def __init__(self, use_orderbook: bool = True, use_technical: bool = True):
        """특징 추출기 초기화"""
        self.orderbook_processor = OrderbookProcessor() if use_orderbook else None
        self.tick_processor = TickDataProcessor()
        self.technical_processor = TechnicalIndicatorProcessor() if use_technical else None
    
    def extract_features(self, tick_data: Dict[str, Any], orderbook_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """통합 특징 추출"""
        all_features = []
        
        # 1. 틱 데이터 특징
        tick_features = self.tick_processor.extract_tick_features(tick_data)
        all_features.extend(tick_features)
        
        # 2. 오더북 특징
        if self.orderbook_processor and orderbook_data:
            orderbook_features = self.orderbook_processor.extract_orderbook_features(orderbook_data)
            all_features.extend(orderbook_features)
        elif self.orderbook_processor:
            # 오더북 데이터가 없는 경우 기본값
            all_features.extend([0.0] * 8)  # 8개 특징
        
        # 3. 기술적 지표
        if self.technical_processor:
            technical_features = self.technical_processor.extract_technical_indicators(tick_data)
            all_features.extend(technical_features)
        
        # 4. Context Vector (System 2에서 전달)
        if "context_vector" in tick_data:
            context = tick_data["context_vector"]
            if isinstance(context, torch.Tensor):
                context = context.cpu().numpy().flatten()
            elif isinstance(context, (list, np.ndarray)):
                context = np.array(context).flatten()
            else:
                context = np.array([0.0] * 10)
            
            # 최대 10차원으로 제한
            if len(context) > 10:
                context = context[:10]
            elif len(context) < 10:
                context = np.pad(context, (0, 10 - len(context)), 'constant')
            
            all_features.extend(context)
        else:
            all_features.extend([0.0] * 10)
        
        return np.array(all_features, dtype=np.float32)

