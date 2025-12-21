"""
시장 소음 필터 (Market Noise Filter)
논문 4.2.2: 횡보장에서의 잦은 손절매 방지
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional


class MarketNoiseFilter:
    """시장 소음 필터 클래스 (논문 4.2.2)"""
    
    def __init__(self,
                 hyperedge_strength_threshold: float = 0.5,
                 transfer_entropy_threshold: float = 2.0,
                 hold_bias: float = 0.3):
        """시장 소음 필터 초기화"""
        self.hyperedge_strength_threshold = hyperedge_strength_threshold
        self.transfer_entropy_threshold = transfer_entropy_threshold
        self.hold_bias = hold_bias
    
    def apply_noise_filter(self,
                          logits: torch.Tensor,
                          hyperedge_strength: Optional[float] = None,
                          transfer_entropy: Optional[float] = None,
                          market_volatility: Optional[float] = None) -> torch.Tensor:
        """
        시장 소음 필터 적용 (논문 4.2.2)
        System 2가 산출한 하이퍼엣지의 연결 강도나 전이 엔트로피 값이 
        특정 임계치 미만일 경우, '관망(Hold)' 행동의 로짓 값에 바이어스 추가
        
        Args:
            logits: 모델 출력 로짓 [batch, num_actions]
            hyperedge_strength: 하이퍼엣지 연결 강도
            transfer_entropy: 전이 엔트로피 값
            market_volatility: 시장 변동성
        Returns:
            filtered_logits: 필터링된 로짓
        """
        filtered_logits = logits.clone()
        
        # 필터링 조건 확인
        should_filter = False
        
        # 1. 하이퍼엣지 연결 강도 확인
        if hyperedge_strength is not None:
            if hyperedge_strength < self.hyperedge_strength_threshold:
                should_filter = True
        
        # 2. 전이 엔트로피 값 확인
        if transfer_entropy is not None:
            if transfer_entropy < self.transfer_entropy_threshold:
                should_filter = True
        
        # 3. 시장 변동성 확인 (횡보장 감지)
        if market_volatility is not None:
            if market_volatility < 0.01:  # 낮은 변동성 = 횡보장
                should_filter = True
        
        # 필터링 적용: '관망(Hold)' 행동의 로짓에 바이어스 추가
        if should_filter:
            # HOLD는 일반적으로 마지막 인덱스 (3: BUY=0, SELL=1, CLOSE=2, HOLD=3)
            # 하위 호환성: output_dim이 3이면 인덱스 2, 4면 인덱스 3
            hold_index = filtered_logits.shape[-1] - 1  # 마지막 인덱스
            filtered_logits[:, hold_index] += self.hold_bias
        
        return filtered_logits
    
    def detect_choppy_market(self,
                            prices: np.ndarray,
                            window: int = 20,
                            threshold: float = 0.02) -> bool:
        """
        횡보장(Choppy Market) 감지
        Args:
            prices: 가격 배열
            window: 윈도우 크기
            threshold: 변동성 임계치
        Returns:
            is_choppy: 횡보장 여부
        """
        if len(prices) < window:
            return False
        
        recent_prices = prices[-window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        
        # 변동성이 낮고 트렌드가 없는 경우 횡보장으로 판단
        if volatility < threshold:
            # 트렌드 확인
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            if abs(trend) < 0.01:  # 거의 변화 없음
                return True
        
        return False
    
    def calculate_hyperedge_strength(self, hyperedge_data: Dict[str, Any]) -> float:
        """하이퍼엣지 연결 강도 계산"""
        weight = hyperedge_data.get("weight", 0.0)
        confidence = hyperedge_data.get("confidence", 0.0)
        strength = weight * confidence
        return strength
    
    def get_transfer_entropy_from_hyperedge(self, hyperedge_data: Dict[str, Any]) -> float:
        """하이퍼엣지에서 전이 엔트로피 값 추출"""
        evidence = hyperedge_data.get("evidence", {})
        transfer_entropy = evidence.get("transfer_entropy", 0.0)
        return transfer_entropy

