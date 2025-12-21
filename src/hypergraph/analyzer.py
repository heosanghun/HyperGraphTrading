"""
하이퍼그래프 분석 도구
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
import networkx as nx

from .structure import HyperNode, HyperEdge, NodeType, RelationType
from .builder import FinancialHypergraph


class HypergraphAnalyzer:
    """하이퍼그래프 분석 클래스"""
    
    def __init__(self, hypergraph: FinancialHypergraph):
        """분석기 초기화"""
        self.hypergraph = hypergraph
    
    def compute_centrality(self, node_id: str) -> Dict[str, float]:
        """노드 중심성 계산"""
        centralities = {}
        
        # Degree centrality (하이퍼그래프 버전)
        degree = len(self.hypergraph.get_neighbors(node_id))
        total_nodes = len(self.hypergraph.nodes)
        centralities['degree'] = degree / max(total_nodes - 1, 1)
        
        # Hyperedge participation
        participation = sum(1 for edge in self.hypergraph.edges.values() 
                          if edge.contains_node(node_id))
        total_edges = len(self.hypergraph.edges)
        centralities['hyperedge_participation'] = participation / max(total_edges, 1)
        
        # Weighted centrality (가중치 고려)
        weighted_degree = 0.0
        for edge in self.hypergraph.edges.values():
            if edge.contains_node(node_id):
                weighted_degree += edge.weight * edge.confidence
        
        centralities['weighted_degree'] = weighted_degree
        
        # NetworkX 기반 중심성 (클리크 그래프 사용)
        if node_id in self.hypergraph.nx_graph:
            nx_centralities = nx.centrality.degree_centrality(self.hypergraph.nx_graph)
            centralities['nx_degree'] = nx_centralities.get(node_id, 0.0)
            
            try:
                betweenness = nx.centrality.betweenness_centrality(self.hypergraph.nx_graph)
                centralities['betweenness'] = betweenness.get(node_id, 0.0)
            except:
                centralities['betweenness'] = 0.0
        
        return centralities
    
    def detect_communities(self) -> Dict[str, int]:
        """커뮤니티 탐지 (시장 섹터 식별)"""
        # NetworkX의 커뮤니티 탐지 알고리즘 사용
        try:
            import networkx.algorithms.community as nx_comm
            
            # 그래프가 너무 작으면 단일 커뮤니티
            if len(self.hypergraph.nx_graph.nodes) < 3:
                return {node: 0 for node in self.hypergraph.nx_graph.nodes()}
            
            # Greedy modularity communities
            communities = nx_comm.greedy_modularity_communities(
                self.hypergraph.nx_graph,
                weight='weight'
            )
            
            # 노드 ID -> 커뮤니티 ID 매핑
            community_map = {}
            for comm_id, comm in enumerate(communities):
                for node_id in comm:
                    community_map[node_id] = comm_id
            
            # 커뮤니티에 속하지 않은 노드 처리
            for node_id in self.hypergraph.nodes.keys():
                if node_id not in community_map:
                    community_map[node_id] = len(communities)
            
            return community_map
            
        except Exception as e:
            print(f"커뮤니티 탐지 오류: {e}")
            # 기본값: 모든 노드를 단일 커뮤니티로
            return {node_id: 0 for node_id in self.hypergraph.nodes.keys()}
    
    def find_influential_nodes(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """영향력이 큰 노드 찾기"""
        node_scores = []
        
        for node_id in self.hypergraph.nodes.keys():
            centralities = self.compute_centrality(node_id)
            # 종합 점수 계산
            score = (
                centralities.get('weighted_degree', 0.0) * 0.4 +
                centralities.get('betweenness', 0.0) * 0.3 +
                centralities.get('hyperedge_participation', 0.0) * 0.3
            )
            node_scores.append((node_id, score))
        
        # 점수 기준 정렬
        node_scores.sort(key=lambda x: x[1], reverse=True)
        return node_scores[:top_k]
    
    def analyze_correlation_structure(self) -> Dict[str, Any]:
        """상관관계 구조 분석"""
        # 노드 타입별 그룹화
        type_groups = defaultdict(list)
        for node_id, node in self.hypergraph.nodes.items():
            type_groups[node.type].append(node_id)
        
        # 타입 간 상관관계 분석
        type_correlations = {}
        for type1, nodes1 in type_groups.items():
            for type2, nodes2 in type_groups.items():
                if type1.value <= type2.value:  # 중복 방지
                    # 두 타입 간의 엣지 찾기
                    edge_count = 0
                    total_weight = 0.0
                    
                    for edge in self.hypergraph.edges.values():
                        node_ids = edge.get_node_ids()
                        has_type1 = any(nid in nodes1 for nid in node_ids)
                        has_type2 = any(nid in nodes2 for nid in node_ids)
                        
                        if has_type1 and has_type2:
                            edge_count += 1
                            total_weight += edge.weight
                    
                    if edge_count > 0:
                        type_correlations[f"{type1.value}_{type2.value}"] = {
                            "edge_count": edge_count,
                            "avg_weight": total_weight / edge_count
                        }
        
        return {
            "type_distribution": {k.value: len(v) for k, v in type_groups.items()},
            "type_correlations": type_correlations,
            "total_nodes": len(self.hypergraph.nodes),
            "total_edges": len(self.hypergraph.edges)
        }
    
    def simulate_influence_propagation(self, 
                                      source_node: str,
                                      max_steps: int = 5,
                                      decay_factor: float = 0.8) -> Dict[str, float]:
        """영향력 전파 시뮬레이션"""
        influence_scores = {source_node: 1.0}
        current_nodes = {source_node}
        
        for step in range(max_steps):
            next_nodes = set()
            next_scores = {}
            
            for node_id in current_nodes:
                current_influence = influence_scores.get(node_id, 0.0)
                
                # 이웃 노드 찾기
                neighbors = self.hypergraph.get_neighbors(node_id)
                
                for neighbor_id in neighbors:
                    # 엣지 가중치 찾기
                    edge_weight = 0.0
                    for edge in self.hypergraph.edges.values():
                        if edge.contains_node(node_id) and edge.contains_node(neighbor_id):
                            edge_weight = max(edge_weight, edge.weight)
                    
                    # 영향력 전파
                    propagated_influence = current_influence * edge_weight * decay_factor
                    
                    if neighbor_id not in influence_scores or \
                       propagated_influence > influence_scores.get(neighbor_id, 0.0):
                        next_scores[neighbor_id] = propagated_influence
                        next_nodes.add(neighbor_id)
            
            # 업데이트
            influence_scores.update(next_scores)
            current_nodes = next_nodes
            
            if not current_nodes:
                break
        
        return influence_scores
    
    def get_market_regime(self) -> str:
        """시장 국면(Regime) 식별 - 개선된 버전"""
        # 1. 가격 데이터 기반 트렌드 분석
        price_trends = []
        volatilities = []
        
        for node_id, node in self.hypergraph.nodes.items():
            if node.type.value == "stock":
                # 가격 데이터 추출
                if 'price_data' in node.features:
                    prices = node.features['price_data']
                elif 'close' in node.features:
                    prices = node.features['close']
                else:
                    continue
                
                if isinstance(prices, list) and len(prices) >= 20:
                    prices_array = np.array(prices)
                    
                    # 트렌드 계산 (최근 20일 평균 vs 전체 평균)
                    recent_avg = np.mean(prices_array[-20:])
                    overall_avg = np.mean(prices_array)
                    trend = (recent_avg - overall_avg) / overall_avg if overall_avg > 0 else 0
                    price_trends.append(trend)
                    
                    # 변동성 계산
                    returns = np.diff(prices_array) / prices_array[:-1]
                    volatility = np.std(returns) if len(returns) > 0 else 0
                    volatilities.append(volatility)
        
        # 2. 상관관계 구조 분석
        correlation_structure = self.analyze_correlation_structure()
        avg_correlation = 0.0
        if correlation_structure["type_correlations"]:
            avg_correlation = np.mean([
                corr["avg_weight"] 
                for corr in correlation_structure["type_correlations"].values()
            ])
        
        # 3. VIX 지수 확인 (거시경제 노드에서)
        vix_value = None
        vix_node = self.hypergraph.get_node("macro_VIX")
        if vix_node and 'value' in vix_node.features:
            vix_value = vix_node.features['value']
        
        # 4. 시장 국면 분류
        avg_trend = np.mean(price_trends) if price_trends else 0.0
        avg_volatility = np.mean(volatilities) if volatilities else 0.0
        
        # 변동성 확대 국면 (Volatile)
        if vix_value and vix_value > 25:
            return "volatile"
        if avg_volatility > 0.03 or avg_correlation > 0.7:
            return "volatile"
        
        # 상승 국면 (Bull)
        if avg_trend > 0.05 and avg_volatility < 0.02:
            return "bull"
        
        # 하락 국면 (Bear)
        if avg_trend < -0.05 and avg_volatility < 0.02:
            return "bear"
        
        # 안정 국면 (Stable)
        if abs(avg_trend) < 0.02 and avg_volatility < 0.015:
            return "stable"
        
        # 기본값
        return "normal"
    
    def reweight_hyperedges_by_regime(self, regime: str) -> Dict[str, float]:
        """시장 국면에 따른 하이퍼엣지 가중치 재할당"""
        reweighting_factors = {}
        
        # 국면별 가중치 증폭 계수
        regime_multipliers = {
            "volatile": {
                "risk_edges": 1.5,  # 리스크 엣지 가중치 증폭
                "correlation_edges": 1.3,  # 상관관계 엣지 증폭
                "normal_edges": 0.8  # 일반 엣지 감소
            },
            "bull": {
                "positive_edges": 1.2,  # 긍정적 엣지 증폭
                "risk_edges": 0.9,  # 리스크 엣지 감소
                "normal_edges": 1.0
            },
            "bear": {
                "risk_edges": 1.4,  # 리스크 엣지 증폭
                "positive_edges": 0.8,  # 긍정적 엣지 감소
                "normal_edges": 1.0
            },
            "stable": {
                "all_edges": 1.0  # 모든 엣지 유지
            },
            "normal": {
                "all_edges": 1.0
            }
        }
        
        multipliers = regime_multipliers.get(regime, {"all_edges": 1.0})
        
        # 각 하이퍼엣지에 대해 재할당
        for edge_id, edge in self.hypergraph.edges.items():
            original_weight = edge.weight
            multiplier = 1.0
            
            # 엣지 타입에 따른 증폭 계수 적용
            if regime == "volatile":
                if edge.relation_type.value in ["influence", "market_impact"]:
                    multiplier = multipliers.get("risk_edges", 1.0)
                elif edge.relation_type.value == "correlation":
                    multiplier = multipliers.get("correlation_edges", 1.0)
                else:
                    multiplier = multipliers.get("normal_edges", 1.0)
            
            elif regime == "bull":
                # 긍정적 엣지 확인 (가중치가 양수인 경우)
                if edge.weight > 0.5:
                    multiplier = multipliers.get("positive_edges", 1.0)
                elif edge.relation_type.value in ["influence", "market_impact"]:
                    multiplier = multipliers.get("risk_edges", 1.0)
                else:
                    multiplier = multipliers.get("normal_edges", 1.0)
            
            elif regime == "bear":
                if edge.relation_type.value in ["influence", "market_impact"]:
                    multiplier = multipliers.get("risk_edges", 1.0)
                elif edge.weight > 0.5:
                    multiplier = multipliers.get("positive_edges", 1.0)
                else:
                    multiplier = multipliers.get("normal_edges", 1.0)
            
            else:  # stable, normal
                multiplier = multipliers.get("all_edges", 1.0)
            
            # 가중치 재할당 (0.0 ~ 1.0 범위 유지)
            new_weight = min(1.0, max(0.0, original_weight * multiplier))
            edge.weight = new_weight
            
            reweighting_factors[edge_id] = {
                "original": original_weight,
                "new": new_weight,
                "multiplier": multiplier
            }
        
        return reweighting_factors

