"""
데이터 전처리 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        """전처리기 초기화"""
        pass
    
    def normalize_timeseries(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """시계열 데이터 정규화"""
        df_normalized = df.copy()
        
        for col in columns:
            if col in df_normalized.columns:
                # Z-score 정규화
                mean = df_normalized[col].mean()
                std = df_normalized[col].std()
                if std > 0:
                    df_normalized[col] = (df_normalized[col] - mean) / std
        
        return df_normalized
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """결측치 처리"""
        df_cleaned = df.copy()
        
        if method == "forward_fill":
            df_cleaned = df_cleaned.ffill()
        elif method == "backward_fill":
            df_cleaned = df_cleaned.bfill()
        elif method == "interpolate":
            df_cleaned = df_cleaned.interpolate()
        else:
            df_cleaned = df_cleaned.fillna(0)
        
        return df_cleaned
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특징 엔지니어링"""
        df_features = df.copy()
        
        # 가격 데이터가 있는 경우
        if "close" in df_features.columns or "Close" in df_features.columns:
            close_col = "close" if "close" in df_features.columns else "Close"
            
            # 이동평균
            df_features["ma_5"] = df_features[close_col].rolling(window=5).mean()
            df_features["ma_20"] = df_features[close_col].rolling(window=20).mean()
            
            # 변동성
            df_features["volatility"] = df_features[close_col].rolling(window=20).std()
            
            # 수익률
            df_features["returns"] = df_features[close_col].pct_change()
            
            # RSI (간단한 구현)
            delta = df_features[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df_features["rsi"] = 100 - (100 / (1 + rs))
        
        return df_features
    
    def prepare_for_hypergraph(self, 
                               price_data: Dict[str, pd.DataFrame],
                               window_size: int = 30) -> Dict[str, np.ndarray]:
        """하이퍼그래프 구축을 위한 데이터 준비"""
        prepared_data = {}
        
        for symbol, df in price_data.items():
            # 최근 window_size일 데이터 추출
            if "close" in df.columns:
                prices = df["close"].tail(window_size).values
            elif "Close" in df.columns:
                prices = df["Close"].tail(window_size).values
            else:
                continue
            
            prepared_data[symbol] = prices
        
        return prepared_data
    
    def preprocess_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """뉴스 데이터 전처리 및 정형화 (1.2)"""
        if news_df.empty:
            return news_df
        
        df = news_df.copy()
        
        # 1. 감성 분석 파싱
        if 'sentiment' in df.columns:
            def parse_sentiment(sentiment_str):
                """sentiment 문자열 파싱"""
                if pd.isna(sentiment_str):
                    return {"class": "neutral", "polarity": 0.0, "subjectivity": 0.0}
                
                if isinstance(sentiment_str, str):
                    try:
                        import ast
                        sentiment_dict = ast.literal_eval(sentiment_str)
                        if isinstance(sentiment_dict, dict):
                            return sentiment_dict
                    except:
                        pass
                
                return {"class": "neutral", "polarity": 0.0, "subjectivity": 0.0}
            
            sentiment_parsed = df['sentiment'].apply(parse_sentiment)
            df['sentiment_class'] = sentiment_parsed.apply(lambda x: x.get('class', 'neutral'))
            df['sentiment_polarity'] = sentiment_parsed.apply(lambda x: x.get('polarity', 0.0))
            df['sentiment_subjectivity'] = sentiment_parsed.apply(lambda x: x.get('subjectivity', 0.0))
        
        # 2. 긴급도 계산 (Urgency Score)
        def calculate_urgency(row):
            """긴급도 점수 계산"""
            urgency = 0.5  # 기본값
            
            # 감성 기반
            if 'sentiment_polarity' in row:
                polarity = abs(row['sentiment_polarity'])
                urgency += polarity * 0.3
            
            # 제목/텍스트 키워드 기반
            title = str(row.get('title', '')).lower()
            text = str(row.get('text', '')).lower()
            
            urgent_keywords = ['urgent', 'breaking', 'crisis', 'crash', 'surge', 'plunge', 
                             'emergency', 'alert', 'warning', 'critical']
            for keyword in urgent_keywords:
                if keyword in title or keyword in text:
                    urgency += 0.2
                    break
            
            # 주제 기반
            subject = str(row.get('subject', '')).lower()
            if 'crisis' in subject or 'war' in subject or 'pandemic' in subject:
                urgency += 0.2
            
            return min(urgency, 1.0)
        
        df['urgency'] = df.apply(calculate_urgency, axis=1)
        
        # 3. 이벤트 타입 추출 (간단한 키워드 기반)
        def extract_event_type(row):
            """이벤트 타입 추출"""
            title = str(row.get('title', '')).lower()
            text = str(row.get('text', '')).lower()
            
            event_types = []
            
            if any(kw in title or kw in text for kw in ['earnings', 'profit', 'revenue', 'quarterly']):
                event_types.append('Earnings')
            if any(kw in title or kw in text for kw in ['merger', 'acquisition', 'deal']):
                event_types.append('M&A')
            if any(kw in title or kw in text for kw in ['rate', 'interest', 'fed', 'federal reserve']):
                event_types.append('Monetary Policy')
            if any(kw in title or kw in text for kw in ['war', 'conflict', 'tension', 'sanction']):
                event_types.append('Geopolitical')
            if any(kw in title or kw in text for kw in ['regulation', 'law', 'policy', 'government']):
                event_types.append('Regulatory')
            if any(kw in title or kw in text for kw in ['inflation', 'cpi', 'gdp', 'economic']):
                event_types.append('Economic Indicator')
            
            return event_types[0] if event_types else 'General News'
        
        df['event_type'] = df.apply(extract_event_type, axis=1)
        
        # 4. 관련 자산 추출 (간단한 심볼 매칭)
        def extract_related_symbols(row):
            """관련 주식 심볼 추출"""
            title = str(row.get('title', '')).upper()
            text = str(row.get('text', '')).upper()
            
            # 주요 주식 심볼 리스트
            common_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
                            'JPM', 'BAC', 'WMT', 'DIS', 'NFLX', 'INTC', 'AMD']
            
            related = []
            for symbol in common_symbols:
                if symbol in title or symbol in text:
                    related.append(symbol)
            
            return related
        
        df['related_symbols'] = df.apply(extract_related_symbols, axis=1)
        
        return df
    
    def calculate_implied_volatility(self,
                                    option_price: float,
                                    underlying_price: float,
                                    strike: float,
                                    time_to_expiry: float,
                                    risk_free_rate: float = 0.02,
                                    option_type: str = "call") -> float:
        """
        내재 변동성(IV) 계산 - Black-Scholes 모델 역산 (논문 5.1.1)
        
        Args:
            option_price: 옵션 가격
            underlying_price: 기초자산 가격
            strike: 행사가
            time_to_expiry: 만기까지 시간 (년 단위)
            risk_free_rate: 무위험 이자율
            option_type: "call" 또는 "put"
        
        Returns:
            내재 변동성 (IV)
        """
        from scipy.stats import norm
        from scipy.optimize import brentq
        
        def black_scholes_price(S, K, T, r, sigma, option_type):
            """Black-Scholes 옵션 가격 계산"""
            if T <= 0:
                return max(S - K, 0) if option_type == "call" else max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == "call":
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return price
        
        def iv_objective(sigma):
            """IV 최적화 목적 함수"""
            bs_price = black_scholes_price(
                underlying_price, strike, time_to_expiry, 
                risk_free_rate, sigma, option_type
            )
            return bs_price - option_price
        
        try:
            # IV 범위 설정 (0.01 ~ 5.0)
            if iv_objective(0.01) * iv_objective(5.0) > 0:
                # 해가 없으면 근사값 반환
                return 0.2  # 기본값
            
            iv = brentq(iv_objective, 0.01, 5.0, maxiter=100)
            return float(iv)
        except Exception as e:
            # 계산 실패 시 기본값 반환
            return 0.2
    
    def calculate_greeks(self,
                        underlying_price: float,
                        strike: float,
                        time_to_expiry: float,
                        risk_free_rate: float,
                        volatility: float,
                        option_type: str = "call") -> Dict[str, float]:
        """
        옵션 그리스 계산 (논문 5.1.1)
        
        Args:
            underlying_price: 기초자산 가격
            strike: 행사가
            time_to_expiry: 만기까지 시간 (년 단위)
            risk_free_rate: 무위험 이자율
            volatility: 변동성 (IV)
            option_type: "call" 또는 "put"
        
        Returns:
            그리스 딕셔너리 (Delta, Gamma, Theta, Vega)
        """
        from scipy.stats import norm
        
        if time_to_expiry <= 0:
            return {
                "delta": 1.0 if (option_type == "call" and underlying_price > strike) else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0
            }
        
        S = underlying_price
        K = strike
        T = time_to_expiry
        r = risk_free_rate
        sigma = volatility
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == "call":
            delta = norm.cdf(d1)
        else:  # put
            delta = -norm.cdf(-d1)
        
        # Gamma (call과 put 동일)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == "call":
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:  # put
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (call과 put 동일)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # 1% 변동성 변화당 가격 변화
        
        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega)
        }
    
    def preprocess_option_data(self, option_df: pd.DataFrame) -> pd.DataFrame:
        """
        옵션 데이터 전처리 (IV 및 그리스 계산 포함)
        """
        if option_df.empty:
            return option_df
        
        df = option_df.copy()
        
        # 필요한 컬럼 확인
        required_cols = ['strike', 'bid', 'ask']
        if not all(col in df.columns for col in required_cols):
            print("[WARNING] 옵션 데이터에 필수 컬럼이 없습니다.")
            return df
        
        # 옵션 가격 (bid-ask 중간값 사용)
        if 'lastPrice' in df.columns:
            df['option_price'] = df['lastPrice'].fillna(
                (df['bid'] + df['ask']) / 2
            )
        else:
            df['option_price'] = (df['bid'] + df['ask']) / 2
        
        # 기초자산 가격 (underlying 컬럼 또는 별도 수집 필요)
        if 'underlying' in df.columns:
            # 실제로는 기초자산 가격을 별도로 가져와야 함
            # 여기서는 간단히 strike 기반으로 근사
            df['underlying_price'] = df['strike'] * 1.0  # 근사값
        else:
            df['underlying_price'] = df['strike'] * 1.0
        
        # 만기까지 시간 계산
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'])
            current_date = pd.Timestamp.now()
            df['time_to_expiry'] = (df['expiration'] - current_date).dt.days / 365.0
            df['time_to_expiry'] = df['time_to_expiry'].clip(lower=0.001)  # 최소값
        else:
            df['time_to_expiry'] = 0.25  # 기본값 3개월
        
        # IV 계산
        df['implied_volatility'] = df.apply(
            lambda row: self.calculate_implied_volatility(
                option_price=row['option_price'],
                underlying_price=row['underlying_price'],
                strike=row['strike'],
                time_to_expiry=row['time_to_expiry'],
                option_type=row.get('option_type', 'call')
            ), axis=1
        )
        
        # 그리스 계산
        greeks_list = []
        for _, row in df.iterrows():
            greeks = self.calculate_greeks(
                underlying_price=row['underlying_price'],
                strike=row['strike'],
                time_to_expiry=row['time_to_expiry'],
                risk_free_rate=0.02,  # 기본값
                volatility=row['implied_volatility'],
                option_type=row.get('option_type', 'call')
            )
            greeks_list.append(greeks)
        
        greeks_df = pd.DataFrame(greeks_list)
        df['delta'] = greeks_df['delta']
        df['gamma'] = greeks_df['gamma']
        df['theta'] = greeks_df['theta']
        df['vega'] = greeks_df['vega']
        
        return df

