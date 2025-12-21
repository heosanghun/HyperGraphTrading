"""
HyperGraphTrading 빠른 시작 예제

이 예제는 HyperGraphTrading 시스템의 기본 사용법을 보여줍니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.hypergraph import FinancialHypergraph
from src.system2 import System2Teacher
from src.system1 import System1Student
from src.integration import SystemIntegrator
from src.trading.backtester import Backtester


def main():
    """빠른 시작 예제"""
    print("="*80)
    print("HyperGraphTrading 빠른 시작 예제")
    print("="*80)
    
    # 1. 데이터 수집
    print("\n[1단계] 데이터 수집")
    collector = DataCollector()
    price_data = collector.collect_price_data(
        symbols=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    print(f"  ✅ {len(price_data)}개 자산 데이터 수집 완료")
    
    # 2. 데이터 전처리
    print("\n[2단계] 데이터 전처리")
    preprocessor = DataPreprocessor()
    processed_data = {}
    for symbol, df in price_data.items():
        df_clean = preprocessor.handle_missing_values(df)
        df_features = preprocessor.engineer_features(df_clean)
        processed_data[symbol] = df_features
    print(f"  ✅ 데이터 전처리 완료")
    
    # 3. 하이퍼그래프 구축
    print("\n[3단계] 하이퍼그래프 구축")
    hypergraph = FinancialHypergraph()
    for symbol, df in processed_data.items():
        from src.hypergraph import HyperNode, NodeType
        node = HyperNode(
            id=symbol,
            type=NodeType.STOCK,
            features={
                "price_data": df["close"].tolist()[-30:] if "close" in df.columns else [],
                "volume": df["volume"].tolist()[-30:] if "volume" in df.columns else []
            }
        )
        hypergraph.add_node(node)
    print(f"  ✅ 하이퍼그래프 구축 완료 ({len(hypergraph.nodes)}개 노드)")
    
    # 4. System 2 (Teacher) 초기화
    print("\n[4단계] System 2 (Teacher) 초기화")
    system2 = System2Teacher(
        hypergraph=hypergraph,
        use_llm=False  # LLM 없이도 동작
    )
    print("  ✅ System 2 초기화 완료")
    
    # 5. System 1 (Student) 초기화
    print("\n[5단계] System 1 (Student) 초기화")
    system1 = System1Student(model_type="simplified")
    print("  ✅ System 1 초기화 완료")
    
    # 6. 시스템 통합
    print("\n[6단계] 시스템 통합")
    integrator = SystemIntegrator(
        system1=system1,
        system2=system2,
        hypergraph=hypergraph
    )
    print("  ✅ 시스템 통합 완료")
    
    # 7. 정책 생성 (System 2)
    print("\n[7단계] System 2 정책 생성")
    policy = system2.generate_policy(
        symbol="AAPL",
        date="2023-12-01",
        use_llm=False
    )
    print(f"  ✅ 정책 생성 완료: {policy.get('decision', 'N/A')}")
    print(f"     신뢰도: {policy.get('confidence', 0):.2f}")
    
    # 8. 실시간 추론 (System 1)
    print("\n[8단계] System 1 실시간 추론")
    tick_data = {
        "close": 150.0,
        "volume": 1000000,
        "prices": [148.0, 149.0, 150.0, 151.0, 150.5]
    }
    decision = system1.infer(tick_data)
    print(f"  ✅ 추론 완료: {decision.get('decision', 'N/A')}")
    print(f"     신뢰도: {decision.get('confidence', 0):.2f}")
    print(f"     추론 시간: {decision.get('inference_time_ms', 0):.2f}ms")
    
    # 9. 백테스팅 (선택적)
    print("\n[9단계] 백테스팅 (선택적)")
    backtester = Backtester(initial_capital=10000.0)
    # 간단한 백테스팅 예제
    for i in range(10):
        price = 150.0 + i * 0.5
        decision = system1.infer({
            "close": price,
            "volume": 1000000,
            "prices": [price-2, price-1, price, price+1, price+0.5]
        })
        backtester.execute_trade(
            decision=decision.get("decision", "HOLD"),
            price=price,
            confidence=decision.get("confidence", 0.5),
            timestamp=f"2023-12-{i+1:02d}"
        )
        backtester.update_equity(price)
    
    metrics = backtester.calculate_metrics()
    print(f"  ✅ 백테스팅 완료")
    print(f"     총 수익률: {metrics.get('total_return', 0):.2%}")
    print(f"     Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"     Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    print("\n" + "="*80)
    print("빠른 시작 예제 완료!")
    print("="*80)
    print("\n더 자세한 사용법은 다음을 참고하세요:")
    print("  - 베이스라인 비교: tests/benchmark/baseline_comparison.py")
    print("  - Ablation Study: tests/benchmark/ablation_study.py")
    print("  - 전체 시스템 실행: scripts/run_full_system.py")


if __name__ == "__main__":
    main()

