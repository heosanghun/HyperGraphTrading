"""
데이터 수집 모듈
"""
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import time


class DataCollector:
    """데이터 수집 클래스"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_price_data(self, 
                          symbols: List[str],
                          start_date: str,
                          end_date: str,
                          interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """주가 데이터 수집"""
        price_data = {}
        
        for symbol in symbols:
            try:
                print(f"수집 중: {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if not df.empty:
                    # 컬럼명 정규화
                    df = df.reset_index()
                    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                    
                    # 저장
                    save_path = self.data_dir / "prices" / f"{symbol}.csv"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(save_path, index=False)
                    
                    price_data[symbol] = df
                    print(f"  [OK] {symbol}: {len(df)}일 데이터 수집 완료")
                else:
                    print(f"  ⚠️ {symbol}: 데이터 없음")
                
                time.sleep(0.5)  # API 호출 제한 방지
                
            except Exception as e:
                print(f"  [ERROR] {symbol}: 오류 - {e}")
        
        return price_data
    
    def collect_news_data(self, symbols: List[str], days: int = 30) -> Dict[str, List[Dict]]:
        """뉴스 데이터 수집 (yfinance 기반, 제한적)"""
        # yfinance는 뉴스 데이터를 직접 제공하지 않음
        # 향후 다른 API로 확장 가능
        news_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # yfinance의 news 속성은 제한적
                news_data[symbol] = []
            except Exception as e:
                print(f"뉴스 수집 오류 ({symbol}): {e}")
        
        return news_data
    
    def load_news_from_csv(self, 
                          csv_path: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """CSV 파일에서 뉴스 데이터 로드 (1.1)"""
        try:
            print(f"[INFO] 뉴스 데이터 로드 중: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # 날짜 컬럼 변환
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            # 날짜 필터링
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['date'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df['date'] <= end_dt]
            
            print(f"[OK] {len(df)}건의 뉴스 데이터 로드 완료")
            return df
            
        except Exception as e:
            print(f"[ERROR] 뉴스 데이터 로드 오류: {e}")
            return pd.DataFrame()
    
    def collect_macro_data(self,
                          indicators: List[str],
                          start_date: str,
                          end_date: str) -> Dict[str, pd.DataFrame]:
        """거시경제 지표 수집 (2.1)"""
        macro_data = {}
        
        # 지표 심볼 매핑
        indicator_symbols = {
            "US10Y": "^TNX",  # 10년 국채 금리
            "US2Y": "^IRX",   # 2년 국채 금리 (근사치)
            "DXY": "DX-Y.NYB",  # 달러 인덱스
            "VIX": "^VIX",    # 변동성 지수
            "WTI": "CL=F",    # 유가
            "GOLD": "GC=F",   # 금
            "SP500": "^GSPC"  # S&P 500 (참고용)
        }
        
        for indicator in indicators:
            if indicator not in indicator_symbols:
                print(f"[WARNING] 알 수 없는 지표: {indicator}")
                continue
            
            symbol = indicator_symbols[indicator]
            try:
                print(f"[INFO] {indicator} 수집 중...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval="1d")
                
                if not df.empty:
                    df = df.reset_index()
                    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                    df['indicator'] = indicator
                    macro_data[indicator] = df
                    print(f"  [OK] {indicator}: {len(df)}일 데이터 수집 완료")
                else:
                    print(f"  ⚠️ {indicator}: 데이터 없음")
                
                time.sleep(0.5)  # API 호출 제한 방지
                
            except Exception as e:
                print(f"  [ERROR] {indicator}: 오류 - {e}")
        
        return macro_data
    
    def collect_option_data(self,
                           underlying_symbol: str,
                           start_date: str,
                           end_date: str,
                           option_type: str = "call") -> pd.DataFrame:
        """
        옵션 데이터 수집 (논문 5.1.1)
        
        Args:
            underlying_symbol: 기초자산 심볼 (예: "AAPL")
            start_date: 시작일
            end_date: 종료일
            option_type: "call" 또는 "put"
        
        Returns:
            옵션 데이터 DataFrame (Strike, Expiration, IV, Greeks 포함)
        """
        try:
            print(f"[INFO] {underlying_symbol} 옵션 데이터 수집 중...")
            ticker = yf.Ticker(underlying_symbol)
            
            # 옵션 만기일 리스트 가져오기
            try:
                expirations = ticker.options
                if not expirations:
                    print(f"  ⚠️ {underlying_symbol}: 옵션 만기일 없음")
                    return pd.DataFrame()
            except Exception as e:
                print(f"  ⚠️ {underlying_symbol}: 옵션 데이터 접근 불가 - {e}")
                return pd.DataFrame()
            
            # 모든 만기일의 옵션 체인 수집
            all_options = []
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            for exp_date in expirations:
                try:
                    exp_dt = pd.to_datetime(exp_date)
                    # 날짜 범위 내의 만기일만 수집
                    if start_dt <= exp_dt <= end_dt:
                        option_chain = ticker.option_chain(exp_date)
                        
                        if option_type == "call":
                            chain_df = option_chain.calls
                        else:
                            chain_df = option_chain.puts
                        
                        if not chain_df.empty:
                            chain_df['expiration'] = exp_date
                            chain_df['underlying'] = underlying_symbol
                            chain_df['option_type'] = option_type
                            all_options.append(chain_df)
                    
                    time.sleep(0.3)  # API 호출 제한 방지
                    
                except Exception as e:
                    print(f"  ⚠️ 만기일 {exp_date} 처리 오류: {e}")
                    continue
            
            if all_options:
                options_df = pd.concat(all_options, ignore_index=True)
                print(f"  [OK] {underlying_symbol}: {len(options_df)}건의 옵션 데이터 수집 완료")
                return options_df
            else:
                print(f"  ⚠️ {underlying_symbol}: 옵션 데이터 없음")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  [ERROR] {underlying_symbol} 옵션 데이터 수집 오류: {e}")
            return pd.DataFrame()
    
    def collect_futures_data(self,
                            symbol: str,
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
        """
        선물 데이터 수집 (논문 5.1.1)
        
        Args:
            symbol: 선물 심볼 (예: "CL=F" (WTI), "GC=F" (Gold), "ES=F" (S&P 500))
            start_date: 시작일
            end_date: 종료일
        
        Returns:
            선물 가격 데이터 DataFrame
        """
        try:
            print(f"[INFO] {symbol} 선물 데이터 수집 중...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if not df.empty:
                df = df.reset_index()
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                df['futures_symbol'] = symbol
                df['instrument_type'] = 'futures'
                
                # 저장
                save_path = self.data_dir / "futures" / f"{symbol}.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path, index=False)
                
                print(f"  [OK] {symbol}: {len(df)}일 선물 데이터 수집 완료")
                return df
            else:
                print(f"  ⚠️ {symbol}: 선물 데이터 없음")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  [ERROR] {symbol} 선물 데이터 수집 오류: {e}")
            return pd.DataFrame()

