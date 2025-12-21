"""
전략 에이전트 (Strategy Agent)
트레이딩 전략 제안
"""
from typing import Dict, List, Any
import numpy as np

from .base_agent import BaseAgent
from ...hypergraph import FinancialHypergraph


class StrategyAgent(BaseAgent):
    """전략 에이전트"""
    
    def __init__(self, hypergraph: FinancialHypergraph):
        """전략 에이전트 초기화"""
        super().__init__(
            name="Strategist",
            role="strategy",
            hypergraph=hypergraph
        )
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """전략 분석"""
        symbol = context.get("symbol", "")
        market_analysis = context.get("market_analysis", {})
        risk_analysis = context.get("risk_analysis", {})
        
        strategy = {
            "symbol": symbol,
            "strategy_type": "MOMENTUM",
            "action": "HOLD",
            "position_size": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "reasoning": []
        }
        
        # 시장 분석 기반 전략
        if market_analysis.get("findings"):
            for finding in market_analysis["findings"]:
                if finding.get("type") == "price_analysis":
                    trend = finding.get("trend", "HOLD")
                    momentum = finding.get("momentum", 0)
                    rsi = finding.get("rsi", 50)
                    ma_short = finding.get("ma_short", 0)
                    ma_long = finding.get("ma_long", 0)
                    
                    # 종합 판단
                    buy_score = 0
                    sell_score = 0
                    
                    # 이동평균 크로스오버
                    if ma_short > ma_long * 1.01:
                        buy_score += 1
                    elif ma_short < ma_long * 0.99:
                        sell_score += 1
                    
                    # 모멘텀
                    if momentum > 0.01:
                        buy_score += 1
                    elif momentum < -0.01:
                        sell_score += 1
                    
                    # RSI
                    if rsi < 40:
                        buy_score += 1
                    elif rsi > 60:
                        sell_score += 1
                    
                    # 최종 결정
                    if buy_score >= 2:
                        strategy["strategy_type"] = "MOMENTUM"
                        strategy["action"] = "BUY"
                        strategy["position_size"] = 0.3
                        strategy["reasoning"].append(f"매수 신호 (점수: {buy_score})")
                    elif sell_score >= 2:
                        strategy["strategy_type"] = "REVERSAL"
                        strategy["action"] = "SELL"
                        strategy["position_size"] = 0.3
                        strategy["reasoning"].append(f"매도 신호 (점수: {sell_score})")
                    else:
                        strategy["action"] = "HOLD"
                        strategy["position_size"] = 0.0
                        strategy["reasoning"].append("신호 혼재, 관망 권장")
        
        # 리스크 분석 기반 조정
        risk_level = risk_analysis.get("risk_level", "LOW")
        risk_recommendation = risk_analysis.get("recommendation", "PROCEED")
        
        if risk_level == "HIGH" or risk_recommendation == "AVOID":
            strategy["action"] = "HOLD"  # 높은 리스크 시 관망
            strategy["position_size"] = 0.0
            strategy["reasoning"].append("높은 리스크 감지, 관망 권장")
        elif risk_level == "MEDIUM" or risk_recommendation == "CAUTION":
            strategy["position_size"] = min(strategy.get("position_size", 0.3), 0.15)  # 포지션 축소
            strategy["reasoning"].append("중간 리스크, 포지션 축소")
        elif risk_level == "LOW":
            # 기존 전략 유지
            pass  # 작은 포지션
            strategy["stop_loss"] = 0.05  # 5% 손절
        elif risk_level == "MEDIUM":
            strategy["position_size"] = 0.3
            strategy["stop_loss"] = 0.03
        else:
            strategy["position_size"] = 0.5
            strategy["stop_loss"] = 0.02
        
        # 하이퍼그래프 기반 전략 조정
        if self.hypergraph:
            node = self.hypergraph.get_node(symbol)
            if node and "price_data" in node.features:
                prices = node.features["price_data"]
                if len(prices) > 0:
                    current_price = prices[-1]
                    strategy["entry_price"] = float(current_price)
                    strategy["stop_loss"] = float(current_price * (1 - strategy["stop_loss"]))
                    strategy["take_profit"] = float(current_price * (1 + strategy["stop_loss"] * 2))
        
        self.update_memory(strategy)
        return strategy
    
    def generate_claim(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """전략 기반 주장 생성"""
        claim = {
            "agent": self.name,
            "type": "strategy_claim",
            "claim": "전략 제안",
            "evidence": evidence,
            "confidence": 0.7
        }
        
        # 근거 기반 전략 결정
        buy_signals = sum(1 for e in evidence if e.get("action") == "BUY")
        sell_signals = sum(1 for e in evidence if e.get("action") == "SELL")
        
        if buy_signals > sell_signals:
            claim["claim"] = f"매수 신호 우세 ({buy_signals} vs {sell_signals})"
            claim["recommendation"] = "BUY"
            claim["confidence"] = min(0.5 + buy_signals * 0.1, 0.9)
        elif sell_signals > buy_signals:
            claim["claim"] = f"매도 신호 우세 ({sell_signals} vs {buy_signals})"
            claim["recommendation"] = "SELL"
            claim["confidence"] = min(0.5 + sell_signals * 0.1, 0.9)
        else:
            claim["claim"] = "신호 혼재, 보류 권장"
            claim["recommendation"] = "HOLD"
            claim["confidence"] = 0.6
        
        return claim

