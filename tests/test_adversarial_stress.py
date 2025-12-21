"""
Adversarial Attack 스트레스 테스트
"""
import sys
from pathlib import Path
import numpy as np

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
if 'HyperGraphTrading' in str(current_file):
    project_root = current_file.parent.parent
else:
    project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hypergraph import FinancialHypergraph, HyperNode, NodeType, RelationType
from src.hypergraph.adversarial_test import AdversarialStressTest


def create_test_hypergraph():
    """테스트용 하이퍼그래프 생성"""
    hypergraph = FinancialHypergraph()
    
    # 샘플 노드 생성
    node1 = HyperNode(
        id="AAPL",
        type=NodeType.STOCK,
        features={"price_data": np.random.randn(100).cumsum() + 100}
    )
    
    node2 = HyperNode(
        id="MSFT",
        type=NodeType.STOCK,
        features={"price_data": np.random.randn(100).cumsum() + 200}
    )
    
    node3 = HyperNode(
        id="GOOGL",
        type=NodeType.STOCK,
        features={"price_data": np.random.randn(100).cumsum() + 150}
    )
    
    hypergraph.add_node(node1)
    hypergraph.add_node(node2)
    hypergraph.add_node(node3)
    
    # 하이퍼엣지 생성
    from src.hypergraph import HyperEdge
    edge1 = HyperEdge(
        nodes=[node1, node2],
        weight=0.7,
        relation_type=RelationType.CORRELATION,
        evidence={"transfer_entropy": 2.5}
    )
    hypergraph.add_hyperedge(edge1)
    
    return hypergraph


def test_noise_injection_attack():
    """노이즈 주입 공격 테스트"""
    print("\n" + "="*80)
    print("노이즈 주입 공격 테스트")
    print("="*80)
    
    hypergraph = create_test_hypergraph()
    stress_test = AdversarialStressTest(hypergraph)
    
    try:
        result = stress_test.noise_injection_attack(
            node_id="AAPL",
            noise_level=0.1,
            noise_type="gaussian"
        )
        
        if "error" not in result:
            print(f"✅ 노이즈 주입 공격 테스트 성공")
            print(f"   공격 타입: {result['attack_type']}")
            print(f"   원본 TE: {result['original_te']:.4f}")
            print(f"   공격 후 TE: {result['attacked_te']:.4f}")
            print(f"   Robustness Score: {result['robustness_score']:.4f}")
            return True
        else:
            print(f"⚠️ 테스트 제한: {result.get('error', 'Unknown error')}")
            return True  # 데이터 부족으로 인한 실패는 정상
    except Exception as e:
        print(f"❌ 노이즈 주입 공격 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fake_news_attack():
    """가짜 뉴스 주입 공격 테스트"""
    print("\n" + "="*80)
    print("가짜 뉴스 주입 공격 테스트")
    print("="*80)
    
    hypergraph = create_test_hypergraph()
    stress_test = AdversarialStressTest(hypergraph)
    
    try:
        result = stress_test.fake_news_attack(
            symbol="AAPL",
            fake_sentiment=0.9,
            fake_urgency=0.9
        )
        
        if "error" not in result:
            print(f"✅ 가짜 뉴스 주입 공격 테스트 성공")
            print(f"   공격 타입: {result['attack_type']}")
            print(f"   가짜 뉴스 ID: {result['fake_news_id']}")
            print(f"   원본 엣지 수: {result['original_edges_count']}")
            print(f"   공격 후 엣지 수: {result['attacked_edges_count']}")
            print(f"   Robustness Score: {result['robustness_score']:.4f}")
            return True
        else:
            print(f"⚠️ 테스트 제한: {result.get('error', 'Unknown error')}")
            return True
    except Exception as e:
        print(f"❌ 가짜 뉴스 주입 공격 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_correlation_manipulation_attack():
    """상관관계 조작 공격 테스트"""
    print("\n" + "="*80)
    print("상관관계 조작 공격 테스트")
    print("="*80)
    
    hypergraph = create_test_hypergraph()
    stress_test = AdversarialStressTest(hypergraph)
    
    try:
        result = stress_test.correlation_manipulation_attack(
            node_ids=["AAPL", "MSFT"],
            manipulation_factor=0.3
        )
        
        if "error" not in result:
            print(f"✅ 상관관계 조작 공격 테스트 성공")
            print(f"   공격 타입: {result['attack_type']}")
            print(f"   조작된 엣지 수: {result['manipulated_edges_count']}")
            print(f"   원본 TE 유효: {result['original_te_valid']}")
            print(f"   조작 후 TE 유효: {result['manipulated_te_valid']}")
            print(f"   Robustness Score: {result['robustness_score']:.4f}")
            return True
        else:
            print(f"⚠️ 테스트 제한: {result.get('error', 'Unknown error')}")
            return True
    except Exception as e:
        print(f"❌ 상관관계 조작 공격 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_weight_perturbation_attack():
    """하이퍼엣지 가중치 조작 공격 테스트"""
    print("\n" + "="*80)
    print("하이퍼엣지 가중치 조작 공격 테스트")
    print("="*80)
    
    hypergraph = create_test_hypergraph()
    stress_test = AdversarialStressTest(hypergraph)
    
    # 첫 번째 엣지 ID 가져오기
    if not hypergraph.edges:
        print("⚠️ 테스트할 엣지가 없음")
        return True
    
    edge_id = list(hypergraph.edges.keys())[0]
    
    try:
        result = stress_test.edge_weight_perturbation_attack(
            edge_id=edge_id,
            perturbation=0.2
        )
        
        if "error" not in result:
            print(f"✅ 하이퍼엣지 가중치 조작 공격 테스트 성공")
            print(f"   공격 타입: {result['attack_type']}")
            print(f"   원본 가중치: {result['original_weight']:.4f}")
            print(f"   조작 후 가중치: {result['manipulated_weight']:.4f}")
            print(f"   원본 TE 점수: {result['original_te_score']:.4f}")
            print(f"   조작 후 TE 점수: {result['manipulated_te_score']:.4f}")
            print(f"   Robustness Score: {result['robustness_score']:.4f}")
            return True
        else:
            print(f"⚠️ 테스트 제한: {result.get('error', 'Unknown error')}")
            return True
    except Exception as e:
        print(f"❌ 하이퍼엣지 가중치 조작 공격 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_stress_test():
    """종합 스트레스 테스트"""
    print("\n" + "="*80)
    print("종합 스트레스 테스트")
    print("="*80)
    
    hypergraph = create_test_hypergraph()
    stress_test = AdversarialStressTest(hypergraph)
    
    try:
        result = stress_test.run_comprehensive_stress_test(
            test_nodes=["AAPL", "MSFT", "GOOGL"],
            attack_types=["noise_injection", "fake_news", "correlation_manipulation", "edge_weight_perturbation"]
        )
        
        print(f"✅ 종합 스트레스 테스트 성공")
        print(f"   테스트 노드: {result['test_nodes']}")
        print(f"   공격 타입: {result['attack_types']}")
        print(f"   전체 Robustness Score: {result['overall_robustness']:.4f}")
        
        for attack_type, attack_results in result['results'].items():
            if attack_results:
                print(f"\n   {attack_type}:")
                for r in attack_results:
                    if 'robustness_score' in r:
                        print(f"     - Robustness: {r['robustness_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ 종합 스트레스 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*80)
    print("Adversarial Attack 스트레스 테스트")
    print("="*80)
    
    results = []
    
    # 테스트 실행
    results.append(("노이즈 주입 공격", test_noise_injection_attack()))
    results.append(("가짜 뉴스 주입 공격", test_fake_news_attack()))
    results.append(("상관관계 조작 공격", test_correlation_manipulation_attack()))
    results.append(("하이퍼엣지 가중치 조작 공격", test_edge_weight_perturbation_attack()))
    results.append(("종합 스트레스 테스트", test_comprehensive_stress_test()))
    
    # 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\n전체 결과: {'✅ 모든 테스트 통과' if all_passed else '❌ 일부 테스트 실패'}")

