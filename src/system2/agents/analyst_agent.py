"""
분석 에이전트 (Analyst Agent)
시장 분석 및 데이터 해석
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from typing import List as ListType

from .base_agent import BaseAgent
from ...hypergraph import FinancialHypergraph


class AnalystAgent(BaseAgent):
    """시장 분석 에이전트"""
    
    def __init__(self, hypergraph: FinancialHypergraph):
        """분석 에이전트 초기화"""
        super().__init__(
            name="Analyst",
            role="market_analysis",
            hypergraph=hypergraph
        )
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """시장 상황 분석"""
        symbol = context.get("symbol", "")
        date = context.get("date", "")
        
        # 하이퍼그래프에서 관련 노드 찾기
        node = self.hypergraph.get_node(symbol) if self.hypergraph else None
        
        analysis = {
            "symbol": symbol,
            "date": date,
            "analysis_type": "market_analysis",
            "findings": [],
            "recommendation": "HOLD"
        }
        
        if node:
            # 가격 데이터 분석
            if "price_data" in node.features:
                prices = node.features["price_data"]
                if len(prices) >= 20:  # 충분한 데이터 필요
                    current_price = prices[-1]
                    avg_price = np.mean(prices)
                    ma_short = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
                    ma_long = np.mean(prices[-20:]) if len(prices) >= 20 else avg_price
                    
                    # 모멘텀 계산
                    momentum = (current_price - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
                    
                    # RSI 계산
                    rsi = self._calculate_rsi(prices[-14:]) if len(prices) >= 14 else 50
                    
                    # 추천 생성
                    recommendation = "HOLD"
                    confidence = 0.5
                    
                    # 모멘텀 + 이동평균 전략
                    if ma_short > ma_long * 1.02 and momentum > 0.01 and rsi < 70:
                        recommendation = "BUY"
                        confidence = min(0.7 + abs(momentum) * 10, 0.9)
                    elif ma_short < ma_long * 0.98 and momentum < -0.01 and rsi > 30:
                        recommendation = "SELL"
                        confidence = min(0.7 + abs(momentum) * 10, 0.9)
                    
                    analysis["findings"].append({
                        "type": "price_analysis",
                        "current_price": current_price,
                        "average_price": avg_price,
                        "ma_short": ma_short,
                        "ma_long": ma_long,
                        "momentum": momentum,
                        "rsi": rsi,
                        "trend": "UP" if current_price > avg_price else "DOWN"
                    })
                    
                    analysis["recommendation"] = recommendation
                    analysis["confidence"] = confidence
            
            # 이웃 노드 분석 (상관관계)
            neighbors = self.hypergraph.get_neighbors(symbol) if self.hypergraph else []
            if neighbors:
                analysis["findings"].append({
                    "type": "correlation_analysis",
                    "correlated_stocks": neighbors,
                    "count": len(neighbors)
                })
        
        # 메모리에 저장
        self.update_memory(analysis)
        
        return analysis
    
    def generate_claim(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """분석 기반 주장 생성"""
        claim = {
            "agent": self.name,
            "type": "analysis_claim",
            "claim": "시장 분석 결과",
            "evidence": evidence,
            "confidence": 0.7
        }
        
        # 근거 분석
        if evidence:
            avg_confidence = np.mean([e.get("confidence", 0) for e in evidence])
            claim["confidence"] = float(avg_confidence)
            
            # 주장 내용 생성
            if avg_confidence > 0.7:
                claim["claim"] = "강한 시장 신호 발견"
            elif avg_confidence > 0.5:
                claim["claim"] = "중간 수준의 시장 신호"
            else:
                claim["claim"] = "약한 시장 신호"
        
        return claim
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

