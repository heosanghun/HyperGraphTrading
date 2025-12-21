"""
동적 하이퍼그래프 업데이트 모듈
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np

from .structure import HyperNode, HyperEdge, NodeType, RelationType
from .builder import FinancialHypergraph


class DynamicHypergraphUpdater:
    """동적 하이퍼그래프 업데이터"""
    
    def __init__(self, hypergraph: FinancialHypergraph, decay_factor: float = 0.95):
        """업데이터 초기화"""
        self.hypergraph = hypergraph
        self.decay_factor = decay_factor
        self.last_update_time: Optional[datetime] = None
    
    def update_weights(self, timestamp: datetime) -> None:
        """가중치 업데이트 (시간 기반 감쇠)"""
        if self.last_update_time is None:
            self.last_update_time = timestamp
            return
        
        # 시간 경과 계산
        time_diff = (timestamp - self.last_update_time).total_seconds() / 3600  # 시간 단위
        
        # 가중치 감쇠 적용
        for edge_id, edge in self.hypergraph.edges.items():
            # 시간에 따른 가중치 감쇠
            decay_steps = int(time_diff)
            for _ in range(decay_steps):
                edge.weight *= self.decay_factor
            
            # 최소 가중치 유지
            edge.weight = max(edge.weight, 0.01)
        
        self.last_update_time = timestamp
    
    def add_new_data(self, 
                    nodes: List[HyperNode],
                    edges: List[HyperEdge],
                    timestamp: datetime) -> None:
        """새 데이터 추가 및 그래프 업데이트"""
        # 새 노드 추가
        for node in nodes:
            if node.id not in self.hypergraph.nodes:
                self.hypergraph.add_node(node)
            else:
                # 기존 노드 업데이트
                existing_node = self.hypergraph.get_node(node.id)
                existing_node.features.update(node.features)
                existing_node.timestamp = timestamp
        
        # 새 엣지 추가 또는 업데이트
        for edge in edges:
            edge_id = self.hypergraph._generate_edge_id(edge)
            
            if edge_id in self.hypergraph.edges:
                # 기존 엣지 업데이트 (가중치 평균 또는 최신 값)
                existing_edge = self.hypergraph.edges[edge_id]
                # 지수 이동 평균으로 업데이트
                alpha = 0.3  # 학습률
                existing_edge.weight = (1 - alpha) * existing_edge.weight + alpha * edge.weight
                existing_edge.confidence = (1 - alpha) * existing_edge.confidence + alpha * edge.confidence
                existing_edge.timestamp = timestamp
            else:
                # 새 엣지 추가
                edge.timestamp = timestamp
                self.hypergraph.add_hyperedge(edge)
        
        # 가중치 업데이트
        self.update_weights(timestamp)
    
    def remove_stale_data(self, max_age_hours: int = 24) -> int:
        """오래된 데이터 제거"""
        if self.last_update_time is None:
            return 0
        
        removed_count = 0
        cutoff_time = self.last_update_time - timedelta(hours=max_age_hours)
        
        # 오래된 노드 제거
        nodes_to_remove = []
        for node_id, node in self.hypergraph.nodes.items():
            if node.timestamp and node.timestamp < cutoff_time:
                # 중요 노드(주식 등)는 제거하지 않음
                if node.type == NodeType.STOCK:
                    continue
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            del self.hypergraph.nodes[node_id]
            self.hypergraph.nx_graph.remove_node(node_id)
            removed_count += 1
        
        # 오래된 엣지 제거
        edges_to_remove = []
        for edge_id, edge in self.hypergraph.edges.items():
            if edge.timestamp and edge.timestamp < cutoff_time:
                # 가중치가 매우 낮은 엣지만 제거
                if edge.weight < 0.1:
                    edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            del self.hypergraph.edges[edge_id]
            removed_count += 1
        
        return removed_count
    
    def detect_structure_change(self, 
                               window_size: int = 10,
                               threshold: float = 0.2) -> bool:
        """구조 변화 감지"""
        # 간단한 구현: 노드/엣지 수 변화 기반
        # 실제로는 더 정교한 방법 사용 가능
        
        if len(self.hypergraph.nodes) < window_size:
            return False
        
        # 최근 변화율 계산 (간단한 휴리스틱)
        # 실제 구현에서는 시계열 분석 필요
        
        return False  # 기본값: 변화 없음
    
    def optimize_graph(self, min_weight: float = 0.1) -> int:
        """그래프 최적화 (낮은 가중치 엣지 제거)"""
        removed_count = 0
        edges_to_remove = []
        
        for edge_id, edge in self.hypergraph.edges.items():
            if edge.weight < min_weight:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            del self.hypergraph.edges[edge_id]
            removed_count += 1
        
        return removed_count

