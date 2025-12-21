"""
LLM 인터페이스
"""
import os
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMInterface(ABC):
    """LLM 인터페이스 베이스 클래스"""
    
    @abstractmethod
    def generate_analysis(self, hypergraph_data: Dict, context: Dict) -> Dict[str, Any]:
        """분석 생성"""
        pass
    
    @abstractmethod
    def evaluate_evidence(self, evidence: List[Dict], claim: Dict) -> Dict[str, Any]:
        """근거 평가"""
        pass


class OpenAIInterface(LLMInterface):
    """OpenAI LLM 인터페이스"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """OpenAI 인터페이스 초기화"""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai 패키지가 설치되지 않았습니다")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def generate_analysis(self, hypergraph_data: Dict, context: Dict) -> Dict[str, Any]:
        """하이퍼그래프 데이터 기반 분석 생성"""
        prompt = self._build_analysis_prompt(hypergraph_data, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 금융 시장 분석 전문가입니다. 하이퍼그래프 데이터를 기반으로 정확하고 근거 있는 분석을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                "analysis": analysis_text,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
        except Exception as e:
            return {
                "analysis": f"분석 생성 오류: {e}",
                "error": str(e)
            }
    
    def evaluate_evidence(self, evidence: List[Dict], claim: Dict) -> Dict[str, Any]:
        """근거 평가"""
        prompt = self._build_evidence_prompt(evidence, claim)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 근거 평가 전문가입니다. 주어진 근거를 객관적으로 평가합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            evaluation_text = response.choices[0].message.content
            
            return {
                "evaluation": evaluation_text,
                "valid": True
            }
        except Exception as e:
            return {
                "evaluation": f"평가 오류: {e}",
                "valid": False,
                "error": str(e)
            }
    
    def _build_analysis_prompt(self, hypergraph_data: Dict, context: Dict) -> str:
        """분석 프롬프트 생성"""
        prompt = f"""다음 하이퍼그래프 데이터를 기반으로 시장을 분석해주세요.

하이퍼그래프 정보:
- 노드 수: {hypergraph_data.get('total_nodes', 0)}
- 엣지 수: {hypergraph_data.get('total_edges', 0)}
- 주요 상관관계: {hypergraph_data.get('type_correlations', {})}

컨텍스트:
- 심볼: {context.get('symbol', 'N/A')}
- 날짜: {context.get('date', 'N/A')}

하이퍼그래프의 구조적 관계를 고려하여 다음을 분석해주세요:
1. 시장 상황 요약
2. 주요 위험 요인
3. 투자 권장사항
"""
        return prompt
    
    def _build_evidence_prompt(self, evidence: List[Dict], claim: Dict) -> str:
        """근거 평가 프롬프트 생성"""
        evidence_summary = "\n".join([
            f"- {e.get('relation', 'N/A')}: 가중치 {e.get('weight', 0):.2f}, 신뢰도 {e.get('confidence', 0):.2f}"
            for e in evidence[:5]  # 상위 5개만
        ])
        
        prompt = f"""다음 주장과 근거를 평가해주세요.

주장: {claim.get('claim', 'N/A')}
신뢰도: {claim.get('confidence', 0):.2f}

근거:
{evidence_summary}

다음을 평가해주세요:
1. 근거의 신뢰성
2. 주장의 타당성
3. 개선 제안
"""
        return prompt


class AnthropicInterface(LLMInterface):
    """Anthropic Claude 인터페이스"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """Anthropic 인터페이스 초기화"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic 패키지가 설치되지 않았습니다")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
    
    def generate_analysis(self, hypergraph_data: Dict, context: Dict) -> Dict[str, Any]:
        """분석 생성 (Claude)"""
        # OpenAI와 유사한 구현
        # 실제 구현은 Anthropic API에 맞게 조정 필요
        return {"analysis": "Claude 분석 (구현 필요)", "model": self.model}
    
    def evaluate_evidence(self, evidence: List[Dict], claim: Dict) -> Dict[str, Any]:
        """근거 평가 (Claude)"""
        return {"evaluation": "Claude 평가 (구현 필요)", "valid": True}


def create_llm_interface(provider: str = "openai", **kwargs) -> LLMInterface:
    """LLM 인터페이스 생성 팩토리"""
    if provider.lower() == "openai":
        return OpenAIInterface(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicInterface(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 LLM 제공자: {provider}")

