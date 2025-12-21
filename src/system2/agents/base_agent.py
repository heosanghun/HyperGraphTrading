"""
멀티 에이전트 시스템 - 베이스 에이전트
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

from ...hypergraph import FinancialHypergraph, HyperNode


class BaseAgent(ABC):
    """에이전트 베이스 클래스"""
    
    def __init__(self, name: str, role: str, hypergraph: Optional[FinancialHypergraph] = None):
        """에이전트 초기화"""
        self.name = name
        self.role = role
        self.hypergraph = hypergraph
        self.memory: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {}
    
    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """상황 분석 (추상 메서드)"""
        pass
    
    @abstractmethod
    def generate_claim(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """주장 생성 (추상 메서드)"""
        pass
    
    def evaluate_claim(self, claim: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """주장 평가"""
        # 기본 구현: 근거 기반 평가
        score = 0.0
        reasoning = []
        
        for ev in evidence:
            if ev.get("confidence", 0) > 0.5:
                score += ev.get("weight", 0) * ev.get("confidence", 0)
                reasoning.append(f"근거: {ev.get('evidence', {})}")
        
        return {
            "score": score / max(len(evidence), 1),
            "reasoning": reasoning,
            "confidence": min(score, 1.0)
        }
    
    def get_evidence_from_hypergraph(self, query: str) -> List[Dict[str, Any]]:
        """하이퍼그래프에서 근거 추출"""
        if self.hypergraph:
            return self.hypergraph.get_evidence(query)
        return []
    
    def update_memory(self, item: Dict[str, Any]) -> None:
        """메모리 업데이트"""
        item["timestamp"] = datetime.now()
        self.memory.append(item)
        
        # 메모리 크기 제한
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
    
    def get_memory(self, limit: int = 10) -> List[Dict[str, Any]]:
        """메모리 조회"""
        return self.memory[-limit:]

