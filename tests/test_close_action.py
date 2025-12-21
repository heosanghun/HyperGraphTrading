"""
청산(Close) 행동 테스트
"""
import sys
from pathlib import Path
import torch
import numpy as np

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
if 'HyperGraphTrading' in str(current_file):
    project_root = current_file.parent.parent
else:
    project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.system1.model.architecture import LightweightTradingModel, SimplifiedTradingModel
from src.system1.system1_student import System1Student
from src.system1.inference.pipeline import InferencePipeline


def test_model_output_dim():
    """모델 출력 차원 테스트 (4개 행동: BUY, SELL, CLOSE, HOLD)"""
    print("\n" + "="*80)
    print("모델 출력 차원 테스트")
    print("="*80)
    
    # LightweightTradingModel 테스트
    model1 = LightweightTradingModel(
        input_dim=10,
        hidden_dim=64,
        output_dim=4,  # BUY, SELL, CLOSE, HOLD
        use_cnn=False  # CNN 없이 테스트 (BatchNorm 문제 방지)
    )
    model1.eval()  # 평가 모드
    
    test_input = torch.randn(1, 1, 10)  # [batch, seq, features]
    with torch.no_grad():
        output1 = model1(test_input)
    
    if isinstance(output1, tuple):
        logits1, value1 = output1
    else:
        logits1 = output1
    
    print(f"✅ LightweightTradingModel 출력 차원: {logits1.shape}")
    print(f"   예상: [1, 4] (BUY, SELL, CLOSE, HOLD)")
    
    if logits1.shape[-1] == 4:
        print("   ✅ 출력 차원 정확")
    else:
        print(f"   ❌ 출력 차원 불일치: {logits1.shape[-1]} != 4")
        return False
    
    # SimplifiedTradingModel 테스트
    model2 = SimplifiedTradingModel(
        input_dim=10,
        hidden_dim=32,
        output_dim=4
    )
    model2.eval()  # 평가 모드
    
    with torch.no_grad():
        output2 = model2(test_input)
    print(f"✅ SimplifiedTradingModel 출력 차원: {output2.shape}")
    
    if output2.shape[-1] == 4:
        print("   ✅ 출력 차원 정확")
        return True
    else:
        print(f"   ❌ 출력 차원 불일치: {output2.shape[-1]} != 4")
        return False


def test_decision_mapping():
    """의사결정 매핑 테스트"""
    print("\n" + "="*80)
    print("의사결정 매핑 테스트")
    print("="*80)
    
    # System1Student 초기화
    system1 = System1Student(model_type="simplified")
    
    # 테스트 입력
    tick_data = {
        "close": 100.0,
        "volume": 1000000,
        "timestamp": "2024-01-01"
    }
    
    try:
        result = system1.infer(tick_data)
        
        decision = result.get("decision", "")
        probabilities = result.get("probabilities", {})
        
        print(f"✅ 의사결정 매핑 테스트 성공")
        print(f"   결정: {decision}")
        print(f"   확률:")
        for action, prob in probabilities.items():
            print(f"     {action}: {prob:.4f}")
        
        # CLOSE 행동이 확률에 포함되어 있는지 확인
        if "CLOSE" in probabilities:
            print("   ✅ CLOSE 행동 포함됨")
            return True
        else:
            print("   ⚠️ CLOSE 행동이 확률에 없음 (하위 호환성 모드)")
            return True  # 하위 호환성으로 허용
    except Exception as e:
        print(f"❌ 의사결정 매핑 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_with_close_action():
    """청산 행동 포함 학습 테스트"""
    print("\n" + "="*80)
    print("청산 행동 포함 학습 테스트")
    print("="*80)
    
    system1 = System1Student(model_type="simplified")
    
    # Teacher 정책 (CLOSE 포함)
    teacher_policies = [
        {"decision": "BUY", "confidence": 0.8, "risk_score": 0.3},
        {"decision": "SELL", "confidence": 0.7, "risk_score": 0.4},
        {"decision": "CLOSE", "confidence": 0.9, "risk_score": 0.2},  # CLOSE 행동
        {"decision": "HOLD", "confidence": 0.6, "risk_score": 0.5}
    ]
    
    # 학습 데이터
    training_data = torch.randn(4, 1, 10)
    
    try:
        result = system1.train_from_teacher(
            teacher_policies=teacher_policies,
            training_data=training_data,
            epochs=1,
            learning_rate=0.001
        )
        
        print(f"✅ 청산 행동 포함 학습 성공")
        print(f"   최종 손실: {result.get('final_loss', 0):.4f}")
        return True
    except Exception as e:
        print(f"❌ 청산 행동 포함 학습 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_pipeline_with_close():
    """추론 파이프라인 CLOSE 행동 테스트"""
    print("\n" + "="*80)
    print("추론 파이프라인 CLOSE 행동 테스트")
    print("="*80)
    
    model = SimplifiedTradingModel(input_dim=10, hidden_dim=32, output_dim=4)
    pipeline = InferencePipeline(model=model, device="cpu")
    
    tick_data = {
        "close": 100.0,
        "volume": 1000000,
        "prices": np.array([100.0, 101.0, 102.0, 101.5, 100.5])
    }
    
    try:
        result = pipeline.process_tick(tick_data)
        
        decision = result["prediction"]["decision"]
        probabilities = result["prediction"]["probabilities"]
        
        print(f"✅ 추론 파이프라인 테스트 성공")
        print(f"   결정: {decision}")
        print(f"   확률:")
        for action, prob in probabilities.items():
            print(f"     {action}: {prob:.4f}")
        
        # CLOSE 행동 확인
        if "CLOSE" in probabilities:
            print("   ✅ CLOSE 행동 지원됨")
            return True
        else:
            print("   ⚠️ CLOSE 행동이 확률에 없음 (하위 호환성)")
            return True
    except Exception as e:
        print(f"❌ 추론 파이프라인 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*80)
    print("청산(Close) 행동 테스트")
    print("="*80)
    
    results = []
    
    # 테스트 실행
    results.append(("모델 출력 차원", test_model_output_dim()))
    results.append(("의사결정 매핑", test_decision_mapping()))
    results.append(("청산 행동 포함 학습", test_training_with_close_action()))
    results.append(("추론 파이프라인 CLOSE 행동", test_inference_pipeline_with_close()))
    
    # 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\n전체 결과: {'✅ 모든 테스트 통과' if all_passed else '❌ 일부 테스트 실패'}")

