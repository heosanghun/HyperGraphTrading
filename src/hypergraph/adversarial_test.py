"""
Adversarial Attack 스트레스 테스트 (논문 3.1.2)
전이 엔트로피 검증 강도 확인
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .builder import FinancialHypergraph
from .transfer_entropy import verify_hyperedge_causality, verify_causality


class AdversarialStressTest:
    """Adversarial Attack 스트레스 테스트 클래스 (논문 3.1.2)"""
    
    def __init__(self, hypergraph: FinancialHypergraph):
        """스트레스 테스트 초기화"""
        self.hypergraph = hypergraph
        self.original_data = {}  # 원본 데이터 백업
    
    def noise_injection_attack(self,
                              node_id: str,
                              noise_level: float = 0.1,
                              noise_type: str = "gaussian") -> Dict[str, Any]:
        """
        노이즈 주입 공격 (논문 3.1.2)
        
        Args:
            node_id: 공격 대상 노드 ID
            noise_level: 노이즈 레벨 (0.0 ~ 1.0)
            noise_type: 노이즈 타입 ("gaussian", "uniform", "outlier")
        
        Returns:
            공격 결과 (원본 TE, 공격 후 TE, Robustness Score)
        """
        node = self.hypergraph.get_node(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}
        
        # 원본 데이터 추출 및 백업
        original_data = None
        if 'price_data' in node.features:
            original_data = np.array(node.features['price_data']).copy()
        elif 'close' in node.features:
            original_data = np.array([node.features['close']]).copy()
        else:
            # 첫 번째 숫자 리스트 찾기
            for key, value in node.features.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        original_data = np.array(value).copy()
                        break
        
        if original_data is None or len(original_data) < 10:
            return {"error": "Insufficient data for attack"}
        
        # 원본 데이터 백업
        self.original_data[node_id] = original_data.copy()
        
        # 노이즈 생성 및 주입
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_level * np.std(original_data), size=original_data.shape)
            attacked_data = original_data + noise
        elif noise_type == "uniform":
            noise = np.random.uniform(-noise_level * np.std(original_data), 
                                    noise_level * np.std(original_data), 
                                    size=original_data.shape)
            attacked_data = original_data + noise
        elif noise_type == "outlier":
            # 이상치 주입
            attacked_data = original_data.copy()
            outlier_indices = np.random.choice(len(attacked_data), 
                                              size=int(len(attacked_data) * noise_level),
                                              replace=False)
            attacked_data[outlier_indices] *= (1 + noise_level * 2)
        else:
            attacked_data = original_data
        
        # 원본 TE 점수 계산
        original_te_scores = []
        neighbors = self.hypergraph.get_neighbors(node_id)
        for neighbor_id in neighbors[:5]:  # 최대 5개 이웃만 테스트
            neighbor_node = self.hypergraph.get_node(neighbor_id)
            if neighbor_node:
                neighbor_data = None
                if 'price_data' in neighbor_node.features:
                    neighbor_data = np.array(neighbor_node.features['price_data'])
                elif 'close' in neighbor_node.features:
                    neighbor_data = np.array([neighbor_node.features['close']])
                
                if neighbor_data is not None and len(neighbor_data) >= 10:
                    min_len = min(len(original_data), len(neighbor_data))
                    is_causal, te_score, z_score = verify_causality(
                        original_data[:min_len],
                        neighbor_data[:min_len],
                        theta=2.0
                    )
                    original_te_scores.append(te_score)
        
        # 공격된 데이터로 TE 점수 계산
        attacked_te_scores = []
        for neighbor_id in neighbors[:5]:
            neighbor_node = self.hypergraph.get_node(neighbor_id)
            if neighbor_node:
                neighbor_data = None
                if 'price_data' in neighbor_node.features:
                    neighbor_data = np.array(neighbor_node.features['price_data'])
                elif 'close' in neighbor_node.features:
                    neighbor_data = np.array([neighbor_node.features['close']])
                
                if neighbor_data is not None and len(neighbor_data) >= 10:
                    min_len = min(len(attacked_data), len(neighbor_data))
                    is_causal, te_score, z_score = verify_causality(
                        attacked_data[:min_len],
                        neighbor_data[:min_len],
                        theta=2.0
                    )
                    attacked_te_scores.append(te_score)
        
        # Robustness Score 계산 (TE 점수 변화율)
        if original_te_scores and attacked_te_scores:
            avg_original_te = np.mean(original_te_scores)
            avg_attacked_te = np.mean(attacked_te_scores)
            
            if avg_original_te > 0:
                robustness_score = 1.0 - abs(avg_original_te - avg_attacked_te) / avg_original_te
            else:
                robustness_score = 0.0
            
            robustness_score = max(0.0, min(1.0, robustness_score))
        else:
            robustness_score = 0.0
            avg_original_te = 0.0
            avg_attacked_te = 0.0
        
        return {
            "attack_type": "noise_injection",
            "noise_level": noise_level,
            "noise_type": noise_type,
            "original_te": float(avg_original_te) if original_te_scores else 0.0,
            "attacked_te": float(avg_attacked_te) if attacked_te_scores else 0.0,
            "robustness_score": float(robustness_score),
            "te_change": float(avg_attacked_te - avg_original_te) if original_te_scores and attacked_te_scores else 0.0
        }
    
    def fake_news_attack(self,
                        symbol: str,
                        fake_sentiment: float = 0.9,
                        fake_urgency: float = 0.9) -> Dict[str, Any]:
        """
        가짜 뉴스 주입 공격 (논문 3.1.2)
        
        Args:
            symbol: 대상 자산 심볼
            fake_sentiment: 가짜 감성 점수
            fake_urgency: 가짜 긴급도
        
        Returns:
            공격 결과
        """
        # 가짜 뉴스 노드 생성
        fake_news_id = f"fake_news_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        fake_news_node = self.hypergraph.create_event_node_from_news({
            'title': f'Fake News: {symbol} Market Crash Imminent',
            'text': f'Fake breaking news about {symbol} market manipulation',
            'sentiment_polarity': fake_sentiment,
            'sentiment_class': 'positive' if fake_sentiment > 0 else 'negative',
            'urgency': fake_urgency,
            'event_type': 'Fake News',
            'related_symbols': [symbol],
            'date': datetime.now()
        })
        
        fake_news_node.id = fake_news_id
        
        # 원본 하이퍼엣지 상태 확인
        symbol_node = self.hypergraph.get_node(symbol)
        if not symbol_node:
            return {"error": f"Symbol {symbol} not found"}
        
        original_edges = []
        for edge_id, edge in self.hypergraph.edges.items():
            if edge.contains_node(symbol):
                original_edges.append({
                    "edge_id": edge_id,
                    "weight": edge.weight,
                    "confidence": edge.confidence
                })
        
        # 가짜 뉴스 노드 추가
        self.hypergraph.add_node(fake_news_node)
        
        # 가짜 뉴스와 자산 간 하이퍼엣지 생성
        fake_edge = self.hypergraph.create_hyperedge_from_news(fake_news_node, [symbol])
        if fake_edge:
            self.hypergraph.add_hyperedge(fake_edge)
        
        # 공격 후 하이퍼엣지 상태 확인
        attacked_edges = []
        for edge_id, edge in self.hypergraph.edges.items():
            if edge.contains_node(symbol):
                attacked_edges.append({
                    "edge_id": edge_id,
                    "weight": edge.weight,
                    "confidence": edge.confidence
                })
        
        # Robustness Score 계산 (엣지 가중치 변화)
        if original_edges:
            avg_original_weight = np.mean([e["weight"] for e in original_edges])
        else:
            avg_original_weight = 0.0
        
        if attacked_edges:
            avg_attacked_weight = np.mean([e["weight"] for e in attacked_edges])
        else:
            avg_attacked_weight = 0.0
        
        if avg_original_weight > 0:
            robustness_score = 1.0 - abs(avg_original_weight - avg_attacked_weight) / avg_original_weight
        else:
            robustness_score = 0.0
        
        robustness_score = max(0.0, min(1.0, robustness_score))
        
        return {
            "attack_type": "fake_news",
            "fake_news_id": fake_news_id,
            "fake_sentiment": fake_sentiment,
            "fake_urgency": fake_urgency,
            "original_edges_count": len(original_edges),
            "attacked_edges_count": len(attacked_edges),
            "original_avg_weight": float(avg_original_weight),
            "attacked_avg_weight": float(avg_attacked_weight),
            "robustness_score": float(robustness_score),
            "weight_change": float(avg_attacked_weight - avg_original_weight)
        }
    
    def correlation_manipulation_attack(self,
                                       node_ids: List[str],
                                       manipulation_factor: float = 0.5) -> Dict[str, Any]:
        """
        상관관계 조작 공격 (논문 3.1.2)
        
        Args:
            node_ids: 조작 대상 노드 ID 리스트
            manipulation_factor: 조작 계수 (0.0 ~ 1.0)
        
        Returns:
            공격 결과
        """
        if len(node_ids) < 2:
            return {"error": "At least 2 nodes required"}
        
        # 원본 하이퍼엣지 가중치 백업
        original_weights = {}
        for edge_id, edge in self.hypergraph.edges.items():
            if any(edge.contains_node(nid) for nid in node_ids):
                original_weights[edge_id] = edge.weight
        
        # 하이퍼엣지 가중치 조작
        manipulated_count = 0
        for edge_id, edge in self.hypergraph.edges.items():
            if any(edge.contains_node(nid) for nid in node_ids):
                # 가중치를 manipulation_factor만큼 조작
                if manipulation_factor > 0:
                    edge.weight = min(1.0, edge.weight * (1 + manipulation_factor))
                else:
                    edge.weight = max(0.0, edge.weight * (1 + manipulation_factor))
                manipulated_count += 1
        
        # 원본 TE 점수 계산
        original_te, _ = verify_hyperedge_causality(
            self.hypergraph,
            node_ids,
            theta=2.0
        )
        
        # 조작 후 TE 점수 계산
        manipulated_te, _ = verify_hyperedge_causality(
            self.hypergraph,
            node_ids,
            theta=2.0
        )
        
        # Robustness Score 계산
        if original_te:
            robustness_score = 1.0 - abs(manipulation_factor)
        else:
            robustness_score = 0.0
        
        return {
            "attack_type": "correlation_manipulation",
            "manipulation_factor": manipulation_factor,
            "manipulated_edges_count": manipulated_count,
            "original_te_valid": original_te,
            "manipulated_te_valid": manipulated_te,
            "robustness_score": float(robustness_score)
        }
    
    def edge_weight_perturbation_attack(self,
                                       edge_id: str,
                                       perturbation: float = 0.2) -> Dict[str, Any]:
        """
        하이퍼엣지 가중치 조작 공격 (논문 3.1.2)
        
        Args:
            edge_id: 조작 대상 엣지 ID
            perturbation: 조작량 (-1.0 ~ 1.0)
        
        Returns:
            공격 결과
        """
        edge = self.hypergraph.edges.get(edge_id)
        if not edge:
            return {"error": f"Edge {edge_id} not found"}
        
        # 원본 가중치 백업
        original_weight = edge.weight
        original_confidence = edge.confidence
        
        # 가중치 조작
        edge.weight = max(0.0, min(1.0, edge.weight + perturbation))
        edge.confidence = max(0.0, min(1.0, edge.confidence + perturbation * 0.5))
        
        # 노드 ID 추출
        node_ids = edge.get_node_ids()
        
        # 원본 TE 점수 계산
        original_te, original_te_score = verify_hyperedge_causality(
            self.hypergraph,
            node_ids,
            theta=2.0
        )
        
        # 조작 후 TE 점수 계산
        manipulated_te, manipulated_te_score = verify_hyperedge_causality(
            self.hypergraph,
            node_ids,
            theta=2.0
        )
        
        # Robustness Score 계산
        if original_te_score > 0:
            robustness_score = 1.0 - abs(perturbation)
        else:
            robustness_score = 0.0
        
        return {
            "attack_type": "edge_weight_perturbation",
            "perturbation": perturbation,
            "original_weight": float(original_weight),
            "manipulated_weight": float(edge.weight),
            "original_te_valid": original_te,
            "original_te_score": float(original_te_score),
            "manipulated_te_valid": manipulated_te,
            "manipulated_te_score": float(manipulated_te_score),
            "robustness_score": float(robustness_score)
        }
    
    def run_comprehensive_stress_test(self,
                                     test_nodes: List[str],
                                     attack_types: List[str] = None) -> Dict[str, Any]:
        """
        종합 스트레스 테스트 실행 (논문 3.1.2)
        
        Args:
            test_nodes: 테스트 대상 노드 ID 리스트
            attack_types: 공격 타입 리스트 (None이면 모두 실행)
        
        Returns:
            종합 테스트 결과
        """
        if attack_types is None:
            attack_types = ["noise_injection", "fake_news", "correlation_manipulation", "edge_weight_perturbation"]
        
        results = {
            "test_nodes": test_nodes,
            "attack_types": attack_types,
            "results": {},
            "overall_robustness": 0.0
        }
        
        robustness_scores = []
        
        for attack_type in attack_types:
            attack_results = []
            
            if attack_type == "noise_injection":
                for node_id in test_nodes[:3]:  # 최대 3개 노드만 테스트
                    result = self.noise_injection_attack(node_id, noise_level=0.1)
                    if "error" not in result:
                        attack_results.append(result)
                        robustness_scores.append(result.get("robustness_score", 0.0))
            
            elif attack_type == "fake_news":
                for node_id in test_nodes[:3]:
                    if self.hypergraph.get_node(node_id):
                        result = self.fake_news_attack(node_id, fake_sentiment=0.9)
                        if "error" not in result:
                            attack_results.append(result)
                            robustness_scores.append(result.get("robustness_score", 0.0))
            
            elif attack_type == "correlation_manipulation":
                if len(test_nodes) >= 2:
                    result = self.correlation_manipulation_attack(test_nodes[:2], manipulation_factor=0.3)
                    if "error" not in result:
                        attack_results.append(result)
                        robustness_scores.append(result.get("robustness_score", 0.0))
            
            elif attack_type == "edge_weight_perturbation":
                # 첫 번째 엣지 선택
                if self.hypergraph.edges:
                    first_edge_id = list(self.hypergraph.edges.keys())[0]
                    result = self.edge_weight_perturbation_attack(first_edge_id, perturbation=0.2)
                    if "error" not in result:
                        attack_results.append(result)
                        robustness_scores.append(result.get("robustness_score", 0.0))
            
            results["results"][attack_type] = attack_results
        
        # 전체 Robustness Score 계산
        if robustness_scores:
            results["overall_robustness"] = float(np.mean(robustness_scores))
        else:
            results["overall_robustness"] = 0.0
        
        return results
    
    def restore_original_data(self, node_id: str) -> bool:
        """원본 데이터 복원"""
        if node_id in self.original_data:
            node = self.hypergraph.get_node(node_id)
            if node and 'price_data' in node.features:
                node.features['price_data'] = self.original_data[node_id].tolist()
                return True
        return False

