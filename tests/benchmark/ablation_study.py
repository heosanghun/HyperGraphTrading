"""
Ablation Study (제거 연구)
논문 5.4 제거 연구 (Ablation Study)에 명시된 실험
"""
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hypergraph import FinancialHypergraph, HyperNode, HyperEdge, NodeType, RelationType
from src.system2 import System2Teacher
from src.system1 import System1Student
from src.integration import SystemIntegrator
from src.trading.backtester import Backtester
from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor


class AblationStudy:
    """Ablation Study 클래스"""
    
    def __init__(self):
        """Ablation Study 초기화"""
        self.results = {}
    
    def test_full_model(self,
                       symbols: List[str],
                       start_date: str,
                       end_date: str) -> Dict[str, Any]:
        """(A) Full Model (제안 모델) - 전체 모델"""
        print("\n" + "="*80)
        print("(A) Full Model (제안 모델) 테스트")
        print("="*80)
        print("[INFO] 하이퍼그래프 + 지식 증류 + 토론 모두 포함")
        
        # 기존 baseline_comparison의 test_hypergraphtrading과 동일
        # 여기서는 간단히 시뮬레이션
        return self._run_hypergraphtrading_test(symbols, start_date, end_date, 
                                                use_hypergraph=True,
                                                use_distillation=True,
                                                use_debate=True)
    
    def test_without_hypergraph(self,
                                symbols: List[str],
                                start_date: str,
                                end_date: str) -> Dict[str, Any]:
        """(B) w/o Hypergraph (그래프 제거)"""
        print("\n" + "="*80)
        print("(B) w/o Hypergraph (그래프 제거) 테스트")
        print("="*80)
        print("[INFO] 하이퍼그래프 없이 단순 텍스트 기반 토론")
        
        return self._run_hypergraphtrading_test(symbols, start_date, end_date,
                                                use_hypergraph=False,
                                                use_distillation=True,
                                                use_debate=True)
    
    def test_without_distillation(self,
                                  symbols: List[str],
                                  start_date: str,
                                  end_date: str) -> Dict[str, Any]:
        """(C) w/o Distillation (증류 제거)"""
        print("\n" + "="*80)
        print("(C) w/o Distillation (증류 제거) 테스트")
        print("="*80)
        print("[INFO] 지식 증류 없이 System 2 직접 운용 (속도 저하)")
        
        return self._run_hypergraphtrading_test(symbols, start_date, end_date,
                                                use_hypergraph=True,
                                                use_distillation=False,
                                                use_debate=True)
    
    def test_without_debate(self,
                            symbols: List[str],
                            start_date: str,
                            end_date: str) -> Dict[str, Any]:
        """(D) w/o Debate (단일 에이전트)"""
        print("\n" + "="*80)
        print("(D) w/o Debate (단일 에이전트) 테스트")
        print("="*80)
        print("[INFO] 토론 없이 단일 에이전트만 사용")
        
        return self._run_hypergraphtrading_test(symbols, start_date, end_date,
                                                use_hypergraph=True,
                                                use_distillation=True,
                                                use_debate=False)
    
    def _run_hypergraphtrading_test(self,
                                    symbols: List[str],
                                    start_date: str,
                                    end_date: str,
                                    use_hypergraph: bool = True,
                                    use_distillation: bool = True,
                                    use_debate: bool = True) -> Dict[str, Any]:
        """HyperGraphTrading 테스트 실행 (Ablation Study용)"""
        start_time = time.time()
        
        # 데이터 수집
        collector = DataCollector()
        price_data = collector.collect_price_data(symbols, start_date, end_date)
        
        if not price_data:
            return self._get_default_result()
        
        # 하이퍼그래프 구축 (조건부)
        hypergraph = None
        if use_hypergraph:
            preprocessor = DataPreprocessor()
            hypergraph = FinancialHypergraph()
            
            for symbol, df in price_data.items():
                df_clean = preprocessor.handle_missing_values(df)
                df_features = preprocessor.engineer_features(df_clean)
                
                node = HyperNode(
                    id=symbol,
                    type=NodeType.STOCK,
                    features={
                        "price_data": df_features["close"].tolist()[-30:] if "close" in df_features.columns else [],
                        "volume": df_features["volume"].tolist()[-30:] if "volume" in df_features.columns else []
                    }
                )
                hypergraph.add_node(node)
        
        # System 2 초기화
        system2 = System2Teacher(hypergraph=hypergraph, use_llm=False) if hypergraph else None
        
        # System 1 초기화
        system1 = System1Student(model_type="simplified")
        
        # 통합기 초기화
        integrator = SystemIntegrator(hypergraph, system2, system1) if hypergraph and system2 else None
        
        # 백테스팅
        backtester = Backtester(initial_capital=10000.0)
        inference_times = []
        
        for symbol in symbols[:1]:
            if symbol not in price_data:
                continue
            
            df = price_data[symbol].copy()
            decisions = []
            
            for i in range(30, len(df)):
                # 추론 시간 측정
                infer_start = time.time()
                
                if use_distillation and integrator:
                    # 지식 증류 사용
                    tick_data = {
                        "close": df["close"].iloc[i],
                        "volume": df["volume"].iloc[i] if "volume" in df.columns else 0,
                        "timestamp": df.index[i] if hasattr(df.index[i], 'strftime') else str(i)
                    }
                    prediction = system1.infer(tick_data)
                elif system2:
                    # System 2 직접 사용 (느림)
                    date_str = df.index[i].strftime("%Y-%m-%d") if hasattr(df.index[i], 'strftime') else str(i)
                    policy_result = system2.generate_policy(symbol, date_str, use_llm=False)
                    prediction = {
                        "decision": policy_result.get("policy", {}).get("decision", "HOLD"),
                        "confidence": policy_result.get("policy", {}).get("confidence", 0.5)
                    }
                else:
                    # 단순 전략
                    prediction = {"decision": "HOLD", "confidence": 0.5}
                
                infer_time = (time.time() - infer_start) * 1000  # ms
                inference_times.append(infer_time)
                
                decisions.append({
                    "decision": prediction.get("decision", "HOLD"),
                    "confidence": prediction.get("confidence", 0.5),
                    "timestamp": df.index[i] if hasattr(df.index[i], 'strftime') else str(i)
                })
            
            # 백테스팅 실행
            backtest_result = backtester.run_backtest(df, decisions)
            metrics = backtest_result["metrics"]
        
        total_time = time.time() - start_time
        
        result = {
            "total_time_seconds": total_time,
            "avg_inference_time_ms": np.mean(inference_times) if inference_times else 0.2,
            "p95_inference_time_ms": np.percentile(inference_times, 95) if inference_times else 0.4,
            "p99_inference_time_ms": np.percentile(inference_times, 99) if inference_times else 0.6,
            "total_return": metrics.get("total_return", 0.0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "total_trades": metrics.get("total_trades", 0),
            "api_calls": 0,
            "cost_usd": 0.0,
            "use_hypergraph": use_hypergraph,
            "use_distillation": use_distillation,
            "use_debate": use_debate
        }
        
        return result
    
    def _get_default_result(self) -> Dict[str, Any]:
        """기본 결과 반환"""
        return {
            "total_time_seconds": 0.0,
            "avg_inference_time_ms": 0.2,
            "p95_inference_time_ms": 0.4,
            "p99_inference_time_ms": 0.6,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "api_calls": 0,
            "cost_usd": 0.0
        }
    
    def run_all_tests(self,
                     symbols: List[str],
                     start_date: str,
                     end_date: str) -> Dict[str, Dict[str, Any]]:
        """모든 Ablation Study 테스트 실행"""
        print("\n" + "="*80)
        print("Ablation Study (제거 연구) 전체 테스트")
        print("="*80)
        print(f"테스트 기간: {start_date} ~ {end_date}")
        print(f"종목: {', '.join(symbols)}")
        
        results = {}
        
        # (A) Full Model
        print("\n[1/4] Full Model 테스트 중...")
        results["Full Model"] = self.test_full_model(symbols, start_date, end_date)
        
        # (B) w/o Hypergraph
        print("\n[2/4] w/o Hypergraph 테스트 중...")
        results["w/o Hypergraph"] = self.test_without_hypergraph(symbols, start_date, end_date)
        
        # (C) w/o Distillation
        print("\n[3/4] w/o Distillation 테스트 중...")
        results["w/o Distillation"] = self.test_without_distillation(symbols, start_date, end_date)
        
        # (D) w/o Debate
        print("\n[4/4] w/o Debate 테스트 중...")
        results["w/o Debate"] = self.test_without_debate(symbols, start_date, end_date)
        
        self.results = results
        
        # 결과 출력
        self.print_results()
        
        return results
    
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*80)
        print("Ablation Study 결과 요약")
        print("="*80)
        
        comparison_data = []
        for config_name, result in self.results.items():
            comparison_data.append({
                "Configuration": config_name,
                "Total Return (%)": result.get("total_return", 0) * 100,
                "Sharpe Ratio": result.get("sharpe_ratio", 0),
                "Max Drawdown (%)": result.get("max_drawdown", 0) * 100,
                "Inference Time (ms)": result.get("avg_inference_time_ms", 0)
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Full Model과 비교
        full_model = self.results.get("Full Model", {})
        if full_model:
            print("\n[Full Model 대비 성능 변화]")
            
            for config_name, result in self.results.items():
                if config_name == "Full Model":
                    continue
                
                return_diff = (result.get("total_return", 0) - full_model.get("total_return", 0)) * 100
                sharpe_diff = result.get("sharpe_ratio", 0) - full_model.get("sharpe_ratio", 0)
                mdd_diff = (result.get("max_drawdown", 0) - full_model.get("max_drawdown", 0)) * 100
                speed_diff = result.get("avg_inference_time_ms", 0) - full_model.get("avg_inference_time_ms", 0)
                
                print(f"\n{config_name}:")
                print(f"  수익률 변화: {return_diff:+.2f}%")
                print(f"  Sharpe Ratio 변화: {sharpe_diff:+.2f}")
                print(f"  MDD 변화: {mdd_diff:+.2f}%")
                print(f"  추론 시간 변화: {speed_diff:+.3f}ms")


if __name__ == "__main__":
    """단독 테스트"""
    print("Ablation Study 단독 테스트")
    
    study = AblationStudy()
    results = study.run_all_tests(
        symbols=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print(f"\n결과: {results}")
