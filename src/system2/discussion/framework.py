"""
근거 중심 토론 프레임워크
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from ..agents.base_agent import BaseAgent


class DiscussionState(Enum):
    """토론 상태"""
    INITIATED = "initiated"
    CLAIM_PRESENTED = "claim_presented"
    COUNTER_PRESENTED = "counter_presented"
    CONSENSUS_REACHED = "consensus_reached"
    DEADLOCK = "deadlock"


class DiscussionFramework:
    """근거 중심 토론 프레임워크"""
    
    def __init__(self, max_rounds: int = 5, consensus_threshold: float = 0.7):
        """토론 프레임워크 초기화"""
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.discussion_log: List[Dict[str, Any]] = []
    
    def initiate_discussion(self, 
                           topic: str,
                           agents: List[BaseAgent],
                           hypergraph) -> Dict[str, Any]:
        """토론 시작"""
        discussion = {
            "topic": topic,
            "agents": [agent.name for agent in agents],
            "state": DiscussionState.INITIATED.value,
            "rounds": [],
            "final_decision": None,
            "consensus_score": 0.0
        }
        
        # 초기 컨텍스트
        context = {
            "topic": topic,
            "hypergraph": hypergraph
        }
        
        # 각 에이전트의 초기 분석
        initial_analyses = {}
        for agent in agents:
            agent.hypergraph = hypergraph
            analysis = agent.analyze(context)
            initial_analyses[agent.name] = analysis
        
        # 토론 라운드 진행
        current_claims = {}
        for round_num in range(self.max_rounds):
            round_result = self._run_round(
                agents=agents,
                round_num=round_num,
                current_claims=current_claims,
                context=context
            )
            
            discussion["rounds"].append(round_result)
            current_claims = round_result.get("claims", {})
            
            # 합의 확인 (Veto Power 포함)
            consensus = self._check_consensus(current_claims, agents=agents, context=context)
            if consensus["reached"]:
                discussion["state"] = DiscussionState.CONSENSUS_REACHED.value
                discussion["final_decision"] = consensus["decision"]
                discussion["consensus_score"] = consensus["score"]
                if consensus.get("vetoed", False):
                    discussion["vetoed"] = True
                    discussion["veto_reason"] = consensus.get("veto_reason", "")
                break
        
        # 합의 도달 실패 시
        if discussion["state"] != DiscussionState.CONSENSUS_REACHED.value:
            discussion["state"] = DiscussionState.DEADLOCK.value
            # 최종 결정: 가장 높은 신뢰도 주장 선택
            if current_claims:
                best_claim = max(
                    current_claims.values(),
                    key=lambda c: c.get("confidence", 0)
                )
                discussion["final_decision"] = best_claim
        
        self.discussion_log.append(discussion)
        return discussion
    
    def _run_round(self,
                  agents: List[BaseAgent],
                  round_num: int,
                  current_claims: Dict[str, Dict],
                  context: Dict[str, Any]) -> Dict[str, Any]:
        """토론 라운드 실행"""
        round_result = {
            "round": round_num,
            "claims": {},
            "counters": {},
            "evidence_used": []
        }
        
        # 각 에이전트가 주장 생성
        for agent in agents:
            # 하이퍼그래프에서 근거 추출
            evidence = agent.get_evidence_from_hypergraph(context.get("topic", ""))
            
            # 주장 생성
            claim = agent.generate_claim(evidence)
            round_result["claims"][agent.name] = claim
            round_result["evidence_used"].extend(evidence)
            
            # 다른 에이전트의 주장에 대한 반박
            for other_agent in agents:
                if other_agent.name != agent.name:
                    if other_agent.name in current_claims:
                        other_claim = current_claims[other_agent.name]
                        counter = agent.evaluate_claim(other_claim, evidence)
                        round_result["counters"][f"{agent.name}_vs_{other_agent.name}"] = counter
        
        return round_result
    
    def _check_consensus(self, 
                        claims: Dict[str, Dict], 
                        agents: Optional[List[BaseAgent]] = None,
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """합의 확인 (Veto Power 포함)"""
        if not claims:
            return {"reached": False, "decision": None, "score": 0.0}
        
        # 주장들을 리스트로 변환
        claims_list = list(claims.values())
        
        # Veto Power 확인 (합의 도출 전)
        if agents:
            risk_agent = None
            for agent in agents:
                if hasattr(agent, 'role') and agent.role == "risk_management":
                    risk_agent = agent
                    break
            
            if risk_agent and hasattr(risk_agent, 'veto'):
                # 임시 합의 도출 (Veto 확인용)
                recommendations = [c.get("recommendation") for c in claims_list if "recommendation" in c]
                if recommendations:
                    proposed_decision = max(set(recommendations), key=recommendations.count)
                    
                    # 리스크 점수 추출
                    risk_score = 0.5
                    for claim in claims_list:
                        if claim.get("agent") == risk_agent.name:
                            if "risk_score" in claim:
                                risk_score = claim["risk_score"]
                            break
                    
                    # Veto 확인
                    veto_result = risk_agent.veto(
                        proposed_decision=proposed_decision,
                        risk_score=risk_score,
                        context=context
                    )
                    
                    if veto_result.get("vetoed", False):
                        # 거부권 발동: 합의 차단
                        return {
                            "reached": False,
                            "decision": "HOLD",
                            "score": 0.0,
                            "vetoed": True,
                            "veto_reason": veto_result.get("reason", "")
                        }
        
        # 주장들의 신뢰도 및 일관성 확인
        confidences = [c.get("confidence", 0) for c in claims_list]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 주장 내용 일치도 확인
        recommendations = [c.get("recommendation") for c in claims_list if "recommendation" in c]
        if recommendations:
            most_common = max(set(recommendations), key=recommendations.count)
            agreement_ratio = recommendations.count(most_common) / len(recommendations)
        else:
            most_common = None
            agreement_ratio = 0.0
        
        # 합의 점수
        consensus_score = (avg_confidence + agreement_ratio) / 2
        
        if consensus_score >= self.consensus_threshold:
            return {
                "reached": True,
                "decision": most_common,
                "score": consensus_score,
                "vetoed": False
            }
        else:
            return {
                "reached": False,
                "decision": most_common,
                "score": consensus_score,
                "vetoed": False
            }
    
    def add_claim(self, agent: BaseAgent, claim: Dict[str, Any], evidence: List[Dict]) -> None:
        """주장 추가"""
        claim["evidence"] = evidence
        claim["timestamp"] = datetime.now()
        self.discussion_log[-1]["rounds"][-1]["claims"][agent.name] = claim
    
    def counter_claim(self, 
                     agent: BaseAgent,
                     claim: Dict[str, Any],
                     counter_evidence: List[Dict]) -> Dict[str, Any]:
        """주장 반박"""
        counter = {
            "agent": agent.name,
            "target_claim": claim,
            "counter_evidence": counter_evidence,
            "timestamp": datetime.now()
        }
        
        # 반박 강도 계산
        counter_strength = sum(e.get("confidence", 0) for e in counter_evidence) / max(len(counter_evidence), 1)
        counter["strength"] = counter_strength
        
        return counter
    
    def reach_consensus(self, 
                      claims: List[Dict[str, Any]], 
                      agents: Optional[List[BaseAgent]] = None,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        합의 도출 (Veto Power 포함) - 논문 3.2.2
        """
        if not claims:
            return {"decision": "HOLD", "confidence": 0.0, "reasoning": "주장 없음"}
        
        # 신뢰도 가중 평균
        total_weight = sum(c.get("confidence", 0) for c in claims)
        if total_weight == 0:
            return {"decision": "HOLD", "confidence": 0.0, "reasoning": "신뢰도 없음"}
        
        # 결정 집계
        decisions = {}
        for claim in claims:
            decision = claim.get("recommendation", "HOLD")
            weight = claim.get("confidence", 0) / total_weight
            decisions[decision] = decisions.get(decision, 0) + weight
        
        # 최종 결정 (가중 평균)
        final_decision = max(decisions.items(), key=lambda x: x[1])
        proposed_decision = final_decision[0]
        proposed_confidence = final_decision[1]
        
        # Veto Power 확인 (리스크 관리자 거부권) - 논문 3.2.2
        if agents:
            risk_agent = None
            for agent in agents:
                if hasattr(agent, 'role') and agent.role == "risk_management":
                    risk_agent = agent
                    break
            
            if risk_agent and hasattr(risk_agent, 'veto'):
                # 리스크 점수 추출
                risk_score = 0.5  # 기본값
                for claim in claims:
                    if claim.get("agent") == risk_agent.name:
                        # 리스크 관련 정보 추출
                        if "risk_score" in claim:
                            risk_score = claim["risk_score"]
                        elif "evidence" in claim:
                            # evidence에서 리스크 점수 찾기
                            for ev in claim["evidence"]:
                                if "risk_score" in ev:
                                    risk_score = ev["risk_score"]
                                    break
                        break
                
                # Veto Power 발동 확인
                veto_result = risk_agent.veto(
                    proposed_decision=proposed_decision,
                    risk_score=risk_score,
                    context=context
                )
                
                if veto_result.get("vetoed", False):
                    # 거부권 발동: 강제로 HOLD 결정
                    return {
                        "decision": "HOLD",
                        "confidence": 0.0,
                        "reasoning": f"리스크 관리자 거부권 발동: {veto_result.get('reason', '')}",
                        "all_decisions": decisions,
                        "vetoed": True,
                        "veto_reason": veto_result.get("reason", ""),
                        "original_decision": proposed_decision
                    }
        
        return {
            "decision": proposed_decision,
            "confidence": proposed_confidence,
            "reasoning": f"가중 평균 기반 합의 (신뢰도: {proposed_confidence:.2f})",
            "all_decisions": decisions,
            "vetoed": False
        }

