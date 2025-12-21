"""
논문 실험 재현 스크립트
논문에 나온 모든 실험을 실제 데이터와 실제 LLM API로 실행
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# .env 파일 로드
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / "env" / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"[확인] .env 파일 로드: {env_path}")
    else:
        load_dotenv()
except ImportError:
    print("[경고] python-dotenv가 설치되지 않았습니다")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 상대 경로로 import
benchmark_dir = Path(__file__).parent / "benchmark"
sys.path.insert(0, str(benchmark_dir))

from baseline_comparison import BaselineComparison
from ablation_study import AblationStudy


class PaperExperimentRunner:
    """논문 실험 실행 클래스"""
    
    def __init__(self):
        """초기화"""
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # 논문 실험 설정
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]  # 논문에서 사용한 종목들
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * 2)  # 2년치 데이터
        
        print("="*80)
        print("논문 실험 재현 스크립트")
        print("="*80)
        print(f"실험 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        print(f"종목: {', '.join(self.symbols)}")
        print("="*80)
    
    def run_all_experiments(self):
        """모든 실험 실행"""
        self.start_time = time.time()
        
        print("\n" + "="*80)
        print("1. 베이스라인 모델 비교 실험")
        print("="*80)
        
        # 1. 베이스라인 모델 비교
        baseline_results = self.run_baseline_comparison()
        self.results["baseline_comparison"] = baseline_results
        
        print("\n" + "="*80)
        print("2. Ablation Study 실험")
        print("="*80)
        
        # 2. Ablation Study
        ablation_results = self.run_ablation_study()
        self.results["ablation_study"] = ablation_results
        
        self.end_time = time.time()
        
        # 결과 저장
        self.save_results()
        
        # 결과 요약 출력
        self.print_summary()
        
        return self.results
    
    def run_baseline_comparison(self):
        """베이스라인 모델 비교 실험"""
        print("\n[베이스라인 모델 비교 실험 시작]")
        print(f"  - 종목: {', '.join(self.symbols)}")
        print(f"  - 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        
        comparator = BaselineComparison()
        results = {}
        
        # 각 베이스라인 모델 테스트
        models = [
            "HyperGraphTrading",  # 제안 모델
            "TradingAgents",
            "TradeMasterEIIE",
            "SingleMarketAgent",
            "TraditionalQuant",
            "BuyHold"
        ]
        
        for model_name in models:
            print(f"\n[{model_name}] 테스트 중...")
            try:
                if model_name == "HyperGraphTrading":
                    result = comparator.test_hypergraphtrading(
                        self.symbols, 
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                elif model_name == "TradingAgents":
                    result = comparator.test_tradingagent_baseline(
                        self.symbols,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                elif model_name == "TradeMasterEIIE":
                    result = comparator.test_trademaster_eiie(
                        self.symbols,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                elif model_name == "SingleMarketAgent":
                    result = comparator.test_single_market_agent(
                        self.symbols,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                elif model_name == "TraditionalQuant":
                    result = comparator.test_traditional_quant(
                        self.symbols,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                elif model_name == "BuyHold":
                    result = comparator.test_buy_hold(
                        self.symbols,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                
                results[model_name] = result
                print(f"  [OK] {model_name} 테스트 완료")
                print(f"     - Sharpe Ratio: {result.get('sharpe_ratio', 0):.4f}")
                print(f"     - Total Return: {result.get('total_return', 0)*100:.2f}%")
                print(f"     - Max Drawdown: {result.get('max_drawdown', 0)*100:.2f}%")
                print(f"     - Inference Time: {result.get('avg_inference_time_ms', 0):.2f}ms")
                
            except Exception as e:
                print(f"  [오류] {model_name} 테스트 실패: {e}")
                results[model_name] = {"error": str(e)}
                import traceback
                traceback.print_exc()
        
        return results
    
    def run_ablation_study(self):
        """Ablation Study 실험"""
        print("\n[Ablation Study 실험 시작]")
        print(f"  - 종목: {', '.join(self.symbols)}")
        print(f"  - 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        
        ablation = AblationStudy()
        results = {}
        
        # Ablation Study 구성
        configurations = [
            "Full Model",
            "w/o Hypergraph",
            "w/o Distillation",
            "w/o Debate"
        ]
        
        for config_name in configurations:
            print(f"\n[{config_name}] 테스트 중...")
            try:
                if config_name == "Full Model":
                    result = ablation.test_full_model(
                        self.symbols,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                elif config_name == "w/o Hypergraph":
                    result = ablation.test_without_hypergraph(
                        self.symbols,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                elif config_name == "w/o Distillation":
                    result = ablation.test_without_distillation(
                        self.symbols,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                elif config_name == "w/o Debate":
                    result = ablation.test_without_debate(
                        self.symbols,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d')
                    )
                
                results[config_name] = result
                print(f"  [OK] {config_name} 테스트 완료")
                print(f"     - Sharpe Ratio: {result.get('sharpe_ratio', 0):.4f}")
                print(f"     - Total Return: {result.get('total_return', 0)*100:.2f}%")
                print(f"     - Max Drawdown: {result.get('max_drawdown', 0)*100:.2f}%")
                print(f"     - Inference Time: {result.get('avg_inference_time_ms', 0):.2f}ms")
                
            except Exception as e:
                print(f"  [오류] {config_name} 테스트 실패: {e}")
                results[config_name] = {"error": str(e)}
                import traceback
                traceback.print_exc()
        
        return results
    
    def save_results(self):
        """결과 저장"""
        # JSON 형식으로 저장
        results_file = "paper_experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[저장] 실험 결과 저장: {results_file}")
        
        # CSV 형식으로 저장 (베이스라인 비교)
        if "baseline_comparison" in self.results:
            baseline_df = pd.DataFrame([
                {
                    "Model": model_name,
                    "Sharpe Ratio": result.get("sharpe_ratio", 0),
                    "Total Return (%)": result.get("total_return", 0) * 100,
                    "Max Drawdown (%)": result.get("max_drawdown", 0) * 100,
                    "Inference Time (ms)": result.get("avg_inference_time_ms", 0),
                    "API Calls": result.get("api_calls", 0),
                    "Cost (USD)": result.get("cost_usd", 0)
                }
                for model_name, result in self.results["baseline_comparison"].items()
            ])
            baseline_df.to_csv("paper_baseline_comparison_results.csv", index=False)
            print(f"[저장] 베이스라인 비교 결과 저장: paper_baseline_comparison_results.csv")
        
        # CSV 형식으로 저장 (Ablation Study)
        if "ablation_study" in self.results:
            ablation_df = pd.DataFrame([
                {
                    "Configuration": config_name,
                    "Sharpe Ratio": result.get("sharpe_ratio", 0),
                    "Total Return (%)": result.get("total_return", 0) * 100,
                    "Max Drawdown (%)": result.get("max_drawdown", 0) * 100,
                    "Inference Time (ms)": result.get("avg_inference_time_ms", 0)
                }
                for config_name, result in self.results["ablation_study"].items()
            ])
            ablation_df.to_csv("paper_ablation_study_results.csv", index=False)
            print(f"[저장] Ablation Study 결과 저장: paper_ablation_study_results.csv")
    
    def print_summary(self):
        """결과 요약 출력"""
        total_time = self.end_time - self.start_time
        
        print("\n" + "="*80)
        print("논문 실험 재현 완료")
        print("="*80)
        print(f"총 실행 시간: {total_time/60:.2f}분 ({total_time:.2f}초)")
        print(f"실험 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        print(f"종목 수: {len(self.symbols)}")
        print("="*80)
        
        # 베이스라인 비교 요약
        if "baseline_comparison" in self.results:
            print("\n[베이스라인 모델 비교 요약]")
            print("-"*80)
            for model_name, result in self.results["baseline_comparison"].items():
                if "error" not in result:
                    print(f"{model_name:20s} | Sharpe: {result.get('sharpe_ratio', 0):6.4f} | "
                          f"Return: {result.get('total_return', 0)*100:6.2f}% | "
                          f"MDD: {result.get('max_drawdown', 0)*100:6.2f}% | "
                          f"Latency: {result.get('avg_inference_time_ms', 0):8.2f}ms")
        
        # Ablation Study 요약
        if "ablation_study" in self.results:
            print("\n[Ablation Study 요약]")
            print("-"*80)
            for config_name, result in self.results["ablation_study"].items():
                if "error" not in result:
                    print(f"{config_name:20s} | Sharpe: {result.get('sharpe_ratio', 0):6.4f} | "
                          f"Return: {result.get('total_return', 0)*100:6.2f}% | "
                          f"MDD: {result.get('max_drawdown', 0)*100:6.2f}% | "
                          f"Latency: {result.get('avg_inference_time_ms', 0):8.2f}ms")
        
        print("\n" + "="*80)
        print("결과 파일:")
        print("  - paper_experiment_results.json")
        print("  - paper_baseline_comparison_results.csv")
        print("  - paper_ablation_study_results.csv")
        print("="*80)


def main():
    """메인 함수"""
    runner = PaperExperimentRunner()
    results = runner.run_all_experiments()
    return results


if __name__ == "__main__":
    main()

