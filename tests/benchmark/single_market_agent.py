"""
TradingAgents의 Market Analyst를 단일 에이전트로 분리
"""
import sys
import os
from pathlib import Path
import time
import numpy as np
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# TradingAgents 경로 추가 (개선된 버전)
import os

# 여러 경로 시도
possible_paths = [
    Path(__file__).parent.parent.parent.parent / "TradingAgents",
    Path(__file__).parent.parent.parent.parent.parent / "TradingAgents",
    Path.cwd() / "TradingAgents",
    Path(os.environ.get("TRADINGAGENTS_PATH", "")),
]

tradingagents_path = None
for path in possible_paths:
    if path and path.exists() and (path / "tradingagents").exists():
        tradingagents_path = path
        break

if tradingagents_path:
    sys.path.insert(0, str(tradingagents_path))
    print(f"[INFO] TradingAgents 경로: {tradingagents_path}")
else:
    print("[WARNING] TradingAgents 경로를 찾을 수 없습니다.")
    print(f"[INFO] 시도한 경로: {[str(p) for p in possible_paths if p]}")

try:
    from langchain_openai import ChatOpenAI
    from tradingagents.agents.analysts.market_analyst import create_market_analyst
    from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators
    from tradingagents.dataflows.config import set_config
    from tradingagents.default_config import DEFAULT_CONFIG
    TRADINGAGENTS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] TradingAgents import 실패: {e}")
    print(f"[INFO] TradingAgents 경로를 확인하거나 환경 변수 TRADINGAGENTS_PATH를 설정하세요.")
    TRADINGAGENTS_AVAILABLE = False


class SingleMarketAgent:
    """TradingAgents의 Market Analyst를 단일 에이전트로 사용"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """단일 Market Agent 초기화"""
        if not TRADINGAGENTS_AVAILABLE:
            raise ImportError("TradingAgents가 설치되지 않았습니다.")
        
        self.config = config or DEFAULT_CONFIG.copy()
        set_config(self.config)
        
        # API 키 검증
        api_key = os.environ.get("OPENAI_API_KEY") or self.config.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다. "
                "환경 변수 OPENAI_API_KEY를 설정하거나 config에 openai_api_key를 추가하세요."
            )
        
        # LLM 초기화 (재시도 로직 포함)
        try:
            if self.config["llm_provider"].lower() == "openai":
                self.llm = ChatOpenAI(
                    model=self.config.get("quick_think_llm", "gpt-4o-mini"),
                    base_url=self.config.get("backend_url"),
                    api_key=api_key,
                    timeout=30.0,  # 타임아웃 설정
                    max_retries=3  # 재시도 횟수
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        except Exception as e:
            print(f"[ERROR] LLM 초기화 실패: {e}")
            raise
        
        # Market Analyst 생성
        try:
            self.market_analyst = create_market_analyst(self.llm)
        except Exception as e:
            print(f"[ERROR] Market Analyst 생성 실패: {e}")
            raise
        
        # Tool nodes
        from langgraph.prebuilt import ToolNode
        self.tool_node = ToolNode([get_stock_data, get_indicators])
    
    def make_decision(self, 
                     symbol: str, 
                     date: str) -> Dict[str, Any]:
        """거래 결정 생성"""
        start_time = time.time()
        
        # 초기 상태 생성
        state = {
            "messages": [{
                "role": "user",
                "content": f"Analyze {symbol} on {date} and provide a trading recommendation."
            }],
            "company_of_interest": symbol,
            "trade_date": date,
            "market_report": "",
        }
        
        # Market Analyst 실행
        try:
            max_iterations = 3  # 최대 반복 횟수
            iteration = 0
            
            while iteration < max_iterations:
                # 1. Market Analyst 분석
                result = self.market_analyst(state)
                state.update(result)
                
                # 2. Tool 호출이 필요한 경우
                if result["messages"] and len(result["messages"]) > 0:
                    last_message = result["messages"][-1]
                    
                    # Tool calls 확인
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        tool_result = self.tool_node.invoke({
                            "messages": result["messages"]
                        })
                        state["messages"].extend(tool_result["messages"])
                        iteration += 1
                        continue
                    else:
                        # Tool 호출이 없으면 완료
                        break
                else:
                    break
            
            # 3. market_report 추출 (여러 위치 확인)
            market_report = (
                state.get("market_report", "") or
                result.get("market_report", "") or
                self._extract_report_from_messages(state.get("messages", []))
            )
            state["market_report"] = market_report
            
            # 4. 결정 추출
            decision = self._extract_decision(market_report)
            confidence = self._calculate_confidence(market_report)
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "decision": decision,
                "confidence": confidence,
                "reasoning": market_report[:500] if market_report else "No report generated",
                "inference_time_ms": inference_time,
                "api_calls": iteration + 1  # LLM 호출 횟수
            }
            
        except Exception as e:
            print(f"[ERROR] SingleMarketAgent 실행 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "decision": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Error: {str(e)}",
                "inference_time_ms": (time.time() - start_time) * 1000,
                "api_calls": 0
            }
    
    def _extract_decision(self, report: str) -> str:
        """보고서에서 결정 추출 (개선된 로직)"""
        if not report:
            return "HOLD"
        
        import re
        report_lower = report.lower()
        
        # 1. FINAL TRANSACTION PROPOSAL 패턴 매칭 (다양한 형식)
        patterns = [
            r"final\s+transaction\s+proposal[:\s]+[*]*\s*(buy|sell|hold)",
            r"recommendation[:\s]+[*]*\s*(buy|sell|hold)",
            r"decision[:\s]+[*]*\s*(buy|sell|hold)",
            r"action[:\s]+[*]*\s*(buy|sell|hold)",
            r"should\s+(buy|sell|hold)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, report_lower, re.IGNORECASE)
            if match:
                decision = match.group(1).upper()
                if decision in ["BUY", "SELL", "HOLD"]:
                    return decision
        
        # 2. 키워드 기반 결정 (가중치 적용)
        bullish_keywords = {
            "buy": 3, "bullish": 2, "upward": 2, "positive": 1, 
            "strong": 2, "growth": 2, "rise": 1, "increase": 1,
            "recommend buying": 3, "suggest buying": 3
        }
        bearish_keywords = {
            "sell": 3, "bearish": 2, "downward": 2, "negative": 1,
            "weak": 2, "decline": 2, "fall": 1, "decrease": 1,
            "recommend selling": 3, "suggest selling": 3
        }
        
        bullish_score = sum(
            weight for keyword, weight in bullish_keywords.items()
            if keyword in report_lower
        )
        bearish_score = sum(
            weight for keyword, weight in bearish_keywords.items()
            if keyword in report_lower
        )
        
        # 3. 점수 기반 결정
        if bullish_score > bearish_score + 2:
            return "BUY"
        elif bearish_score > bullish_score + 2:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_report_from_messages(self, messages: List) -> str:
        """메시지에서 보고서 추출"""
        report = ""
        for msg in reversed(messages):  # 최신 메시지부터
            content = ""
            if isinstance(msg, dict):
                content = msg.get("content", "")
            elif hasattr(msg, 'content'):
                content = msg.content
            
            if "FINAL TRANSACTION PROPOSAL" in content or len(content) > 100:
                report = content
                break
        
        return report
    
    def _calculate_confidence(self, report: str) -> float:
        """보고서 기반 신뢰도 계산"""
        if not report:
            return 0.5
        
        # FINAL TRANSACTION PROPOSAL이 있으면 높은 신뢰도
        if "final transaction proposal" in report.lower():
            return 0.8
        
        # 키워드 밀도 기반 신뢰도
        keywords_found = sum(1 for word in ["buy", "sell", "hold", "recommend", "suggest"] 
                            if word in report.lower())
        confidence = min(0.5 + keywords_found * 0.1, 0.9)
        
        return confidence


def test_single_market_agent():
    """테스트 함수"""
    if not TRADINGAGENTS_AVAILABLE:
        print("[SKIP] TradingAgents가 설치되지 않아 테스트를 건너뜁니다.")
        return
    
    try:
        agent = SingleMarketAgent()
        result = agent.make_decision("AAPL", "2024-05-10")
        print(f"결정: {result['decision']}")
        print(f"신뢰도: {result['confidence']}")
        print(f"추론 시간: {result['inference_time_ms']:.2f}ms")
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")


if __name__ == "__main__":
    test_single_market_agent()

