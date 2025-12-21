"""
정책 추출 모듈
토론 결과에서 트레이딩 정책 추출
"""
from typing import Dict, List, Any
from datetime import datetime
import json
from collections import Counter


class PolicyExtractor:
    """정책 추출기"""
    
    def __init__(self):
        """정책 추출기 초기화"""
        pass
    
    def extract_policy(self, discussion_result: Dict[str, Any]) -> Dict[str, Any]:
        """토론 결과에서 정책 추출"""
        # 최종 결정 추출 (개선)
        final_decision = discussion_result.get("final_decision", {})
        
        # final_decision이 dict인 경우
        if isinstance(final_decision, dict):
            decision = final_decision.get("decision", "HOLD")
            confidence = final_decision.get("confidence", discussion_result.get("consensus_score", 0.0))
        else:
            decision = "HOLD"
            confidence = discussion_result.get("consensus_score", 0.0)
        
        # 토론 라운드에서 추천 수집
        recommendations = []
        for round_data in discussion_result.get("rounds", []):
            for claim in round_data.get("claims", {}).values():
                if "recommendation" in claim:
                    recommendations.append(claim["recommendation"])
                elif "action" in claim:
                    recommendations.append(claim["action"])
        
        # 추천이 있으면 사용
        if recommendations:
            # 가장 많이 나온 추천 선택
            rec_counter = Counter(recommendations)
            most_common = rec_counter.most_common(1)
            if most_common:
                candidate = most_common[0][0]
                # 유효한 결정인지 확인 (논문 4.2.2: BUY, SELL, CLOSE, HOLD)
                valid_decisions = ["BUY", "SELL", "CLOSE", "HOLD"]
                if candidate in valid_decisions:
                    decision = candidate
                    confidence = max(confidence, most_common[0][1] / len(recommendations))
                else:
                    # 유효하지 않은 경우, 다른 추천 찾기
                    for rec, count in rec_counter.most_common():
                        if rec in valid_decisions:
                            decision = rec
                            confidence = max(confidence, count / len(recommendations))
                            break
                    # 여전히 유효한 결정이 없으면 HOLD
                    if decision not in valid_decisions:
                        decision = "HOLD"
        
        policy = {
            "symbol": discussion_result.get("topic", ""),
            "decision": decision,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "reasoning": [],
            "parameters": {}
        }
        
        # 토론 라운드에서 추론 경로 추출
        for round_data in discussion_result.get("rounds", []):
            for claim in round_data.get("claims", {}).values():
                if "reasoning" in claim:
                    policy["reasoning"].extend(claim["reasoning"])
                if "recommendation" in claim:
                    policy["reasoning"].append(f"{claim['agent']}: {claim['recommendation']}")
        
        # 최종 결정 기반 파라미터 설정
        final_decision = discussion_result.get("final_decision", {})
        if isinstance(final_decision, dict):
            policy["parameters"] = final_decision.get("parameters", {})
        elif isinstance(final_decision, str):
            # 문자열인 경우 기본 파라미터
            if final_decision == "BUY":
                policy["parameters"] = {
                    "action": "BUY",
                    "position_size": 0.3,
                    "stop_loss": 0.02,
                    "take_profit": 0.04
                }
            elif final_decision == "SELL":
                policy["parameters"] = {
                    "action": "SELL",
                    "position_size": 0.3,
                    "stop_loss": 0.02,
                    "take_profit": 0.04
                }
            else:
                policy["parameters"] = {
                    "action": "HOLD",
                    "position_size": 0.0
                }
        
        return policy
    
    def validate_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """정책 검증"""
        validation = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # 필수 필드 확인
        required_fields = ["symbol", "decision", "confidence"]
        for field in required_fields:
            if field not in policy:
                validation["valid"] = False
                validation["errors"].append(f"필수 필드 누락: {field}")
        
        # 신뢰도 확인
        if policy.get("confidence", 0) < 0.5:
            validation["warnings"].append("낮은 신뢰도 정책")
        
        # 결정 타입 확인 (논문 4.2.2: BUY, SELL, CLOSE, HOLD)
        valid_decisions = ["BUY", "SELL", "CLOSE", "HOLD"]
        if policy.get("decision") not in valid_decisions:
            validation["valid"] = False
            validation["errors"].append(f"잘못된 결정: {policy.get('decision')}")
        
        return validation
    
    def save_policy(self, policy: Dict[str, Any], filepath: str) -> None:
        """정책 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(policy, f, ensure_ascii=False, indent=2)
    
    def load_policy(self, filepath: str) -> Dict[str, Any]:
        """정책 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

