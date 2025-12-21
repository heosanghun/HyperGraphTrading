"""
하이퍼그래프 데이터 구조 정의
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class NodeType(Enum):
    """노드 타입 정의"""
    STOCK = "stock"
    NEWS = "news"
    SENTIMENT = "sentiment"
    ECONOMIC = "economic"
    SECTOR = "sector"
    INDUSTRY = "industry"
    OPTION = "option"  # 옵션 노드 (논문 5.1.1)
    FUTURES = "futures"  # 선물 노드 (논문 5.1.1)


class RelationType(Enum):
    """관계 타입 정의"""
    CORRELATION = "correlation"
    CAUSALITY = "causality"
    INFLUENCE = "influence"
    SECTOR_RELATION = "sector_relation"
    MARKET_IMPACT = "market_impact"


@dataclass
class HyperNode:
    """하이퍼그래프 노드"""
    id: str
    type: NodeType
    features: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, HyperNode):
            return self.id == other.id
        return False


@dataclass
class HyperEdge:
    """하이퍼그래프 엣지 (하이퍼엣지)"""
    nodes: List[HyperNode]
    weight: float
    relation_type: RelationType
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """엣지 검증"""
        if len(self.nodes) < 2:
            raise ValueError("하이퍼엣지는 최소 2개 이상의 노드를 포함해야 합니다")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("가중치는 0.0과 1.0 사이여야 합니다")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("신뢰도는 0.0과 1.0 사이여야 합니다")
    
    def get_node_ids(self) -> List[str]:
        """노드 ID 리스트 반환"""
        return [node.id for node in self.nodes]
    
    def contains_node(self, node_id: str) -> bool:
        """특정 노드 포함 여부 확인"""
        return any(node.id == node_id for node in self.nodes)

