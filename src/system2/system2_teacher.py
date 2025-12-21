"""
System 2 (Teacher) 메인 클래스
하이퍼그래프 기반 근거 중심 토론 시스템
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from .agents.analyst_agent import AnalystAgent
from .agents.risk_agent import RiskAgent
from .agents.strategy_agent import StrategyAgent
from .discussion.framework import DiscussionFramework
from .llm.interface import create_llm_interface
from .policy.extractor import PolicyExtractor
from ..hypergraph import FinancialHypergraph, NodeType, NodeType


class System2Teacher:
    """System 2 (Teacher) 시스템"""
    
    def __init__(self, 
                 hypergraph: FinancialHypergraph,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 use_llm: bool = False):
        """System 2 초기화"""
        self.hypergraph = hypergraph
        
        # 에이전트 생성
        self.analyst = AnalystAgent(hypergraph)
        self.risk_manager = RiskAgent(hypergraph)
        self.strategist = StrategyAgent(hypergraph)
        self.agents = [self.analyst, self.risk_manager, self.strategist]
        
        # 토론 프레임워크
        self.discussion_framework = DiscussionFramework(
            max_rounds=5,
            consensus_threshold=0.7
        )
        
        # LLM 인터페이스 (선택적)
        self.use_llm = use_llm
        if use_llm:
            try:
                self.llm = create_llm_interface(provider=llm_provider, model=llm_model)
            except Exception as e:
                print(f"LLM 초기화 실패 (선택적): {e}")
                self.llm = None
        else:
            self.llm = None
        
        # 정책 추출기
        self.policy_extractor = PolicyExtractor()
    
    def generate_policy(self, 
                       symbol: str,
                       date: str,
                       use_llm: bool = False) -> Dict[str, Any]:
        """정책 생성"""
        print(f">>> System 2: {symbol} 정책 생성 시작 ({date})")
        
        # 컨텍스트 준비
        context = {
            "symbol": symbol,
            "date": date,
            "hypergraph": self.hypergraph
        }
        
        # 각 에이전트 분석
        print("  - 에이전트 분석 중...")
        market_analysis = self.analyst.analyze(context)
        risk_analysis = self.risk_manager.analyze(context)
        
        context["market_analysis"] = market_analysis
        context["risk_analysis"] = risk_analysis
        
        strategy_analysis = self.strategist.analyze(context)
        context["strategy_analysis"] = strategy_analysis
        
        # 토론 시작
        print("  - 토론 진행 중...")
        topic = f"{symbol} 트레이딩 결정 ({date})"
        discussion = self.discussion_framework.initiate_discussion(
            topic=topic,
            agents=self.agents,
            hypergraph=self.hypergraph
        )
        
        # Veto Power 정보를 context에 추가
        if discussion.get("vetoed", False):
            context["vetoed"] = True
            context["veto_reason"] = discussion.get("veto_reason", "")
        
        # LLM 분석 (선택적)
        if use_llm and self.llm:
            print("  - LLM 분석 중...")
            hypergraph_data = {
                "total_nodes": len(self.hypergraph.nodes),
                "total_edges": len(self.hypergraph.edges),
                "type_correlations": {}
            }
            llm_analysis = self.llm.generate_analysis(hypergraph_data, context)
            discussion["llm_analysis"] = llm_analysis
        
        # 정책 추출
        print("  - 정책 추출 중...")
        policy = self.policy_extractor.extract_policy(discussion)
        
        # 정책 검증
        validation = self.policy_extractor.validate_policy(policy)
        policy["validation"] = validation
        
        if not validation["valid"]:
            print(f"  [WARNING] 정책 검증 실패: {validation['errors']}")
        
        print(f"<<< System 2: 정책 생성 완료 (결정: {policy['decision']}, 신뢰도: {policy['confidence']:.2f})")
        
        return {
            "policy": policy,
            "discussion": discussion,
            "analyses": {
                "market": market_analysis,
                "risk": risk_analysis,
                "strategy": strategy_analysis
            }
        }
    
    def analyze_news_for_policy(self, symbol: str, date: str) -> Dict:
        """뉴스 데이터를 분석하여 정책 수립에 활용 (1.5)"""
        print(f">>> System 2: {symbol} 뉴스 분석 시작 ({date})")
        
        # 해당 날짜의 뉴스 조회
        news_nodes = []
        for node_id, node in self.hypergraph.nodes.items():
            if node.type == NodeType.NEWS:
                # 날짜 매칭
                if node.timestamp:
                    node_date = node.timestamp.strftime("%Y-%m-%d")
                    if node_date == date:
                        # 관련 심볼 확인
                        related_symbols = node.features.get('related_symbols', [])
                        if symbol in related_symbols or symbol in node.features.get('text', '').upper():
                            news_nodes.append(node)
        
        if not news_nodes:
            return {
                "news_count": 0,
                "sentiment_avg": 0.0,
                "urgency_avg": 0.5,
                "hypotheses": []
            }
        
        # 뉴스 분석
        sentiment_scores = [n.features.get('sentiment', 0.0) for n in news_nodes]
        urgency_scores = [n.features.get('urgency', 0.5) for n in news_nodes]
        
        # LLM이 뉴스를 독해하여 인과관계 가설 생성
        hypotheses = []
        if self.llm:
            for news_node in news_nodes[:5]:  # 최대 5개만 분석
                title = news_node.features.get('title', '')
                text = news_node.features.get('text', '')
                
                # 간단한 가설 생성 (LLM 없이도 동작)
                hypothesis = {
                    "event": news_node.features.get('event_type', 'General News'),
                    "description": title[:100],
                    "impact": "positive" if news_node.features.get('sentiment', 0) > 0 else "negative",
                    "confidence": news_node.features.get('urgency', 0.5)
                }
                hypotheses.append(hypothesis)
        else:
            # LLM 없이 간단한 가설 생성
            for news_node in news_nodes[:5]:
                hypothesis = {
                    "event": news_node.features.get('event_type', 'General News'),
                    "description": news_node.features.get('title', '')[:100],
                    "impact": "positive" if news_node.features.get('sentiment', 0) > 0 else "negative",
                    "confidence": news_node.features.get('urgency', 0.5)
                }
                hypotheses.append(hypothesis)
        
        result = {
            "news_count": len(news_nodes),
            "sentiment_avg": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0,
            "urgency_avg": sum(urgency_scores) / len(urgency_scores) if urgency_scores else 0.5,
            "hypotheses": hypotheses
        }
        
        print(f"  [OK] {len(news_nodes)}건의 뉴스 분석 완료")
        return result
    
    def analyze_macro_for_policy(self, symbol: str, date: str) -> Dict:
        """거시경제 지표를 분석하여 정책 수립에 활용 (2.5)"""
        print(f">>> System 2: {symbol} 거시경제 지표 분석 시작 ({date})")
        
        # 해당 날짜의 거시경제 지표 조회
        macro_nodes = []
        target_date = pd.to_datetime(date)
        
        for node_id, node in self.hypergraph.nodes.items():
            if node.type == NodeType.ECONOMIC:
                # 날짜 매칭 (근사치)
                if node.timestamp:
                    date_diff = abs((node.timestamp - target_date).days)
                    if date_diff <= 1:  # 1일 이내
                        macro_nodes.append(node)
        
        if not macro_nodes:
            return {
                "macro_count": 0,
                "indicators": {},
                "market_impact": "neutral"
            }
        
        # 거시경제 지표 분석
        indicators = {}
        for macro_node in macro_nodes:
            indicator = macro_node.features.get('indicator', '')
            value = macro_node.features.get('value', 0)
            trend = macro_node.features.get('trend', 'Stable')
            indicators[indicator] = {
                "value": value,
                "trend": trend
            }
        
        # 시장 전체 영향 분석
        market_impact = "neutral"
        positive_count = sum(1 for ind in indicators.values() if ind['trend'] == 'Upward')
        negative_count = sum(1 for ind in indicators.values() if ind['trend'] == 'Downward')
        
        if positive_count > negative_count:
            market_impact = "positive"
        elif negative_count > positive_count:
            market_impact = "negative"
        
        result = {
            "macro_count": len(macro_nodes),
            "indicators": indicators,
            "market_impact": market_impact
        }
        
        print(f"  [OK] {len(macro_nodes)}개의 거시경제 지표 분석 완료")
        return result
    
    def extract_causal_paths(self, symbol: str, date: str, max_paths: int = 10) -> List[Dict[str, Any]]:
        """하이퍼그래프에서 인과 경로 추출 (Reasoning Distillation용)"""
        causal_paths = []
        
        # 해당 심볼과 관련된 하이퍼엣지 찾기
        symbol_node = self.hypergraph.get_node(symbol)
        if not symbol_node:
            return causal_paths
        
        # 관련 엣지 수집
        relevant_edges = []
        for edge_id, edge in self.hypergraph.edges.items():
            if edge.contains_node(symbol):
                relevant_edges.append(edge)
        
        # 전이 엔트로피 점수 기준 정렬
        relevant_edges.sort(
            key=lambda e: e.evidence.get("transfer_entropy", 0),
            reverse=True
        )
        
        # 상위 경로 추출
        for edge in relevant_edges[:max_paths]:
            node_ids = edge.get_node_ids()
            
            # 경로 정보 구성
            path_info = {
                "nodes": node_ids,
                "edges": [edge_id for edge_id, e in self.hypergraph.edges.items() if e == edge],
                "weights": [edge.weight],
                "transfer_entropy": [edge.evidence.get("transfer_entropy", 0.0)],
                "confidence": edge.confidence,
                "relation_type": edge.relation_type.value
            }
            
            causal_paths.append(path_info)
        
        return causal_paths
    
    def extract_risk_adjusted_return(self, policy: Dict[str, Any]) -> float:
        """위험 조정 기대 수익률 추출 (Value Distillation용)"""
        # 정책에서 리스크 조정 수익률 계산
        decision = policy.get("decision", "HOLD")
        confidence = policy.get("confidence", 0.5)
        risk_score = policy.get("risk_score", 0.5)
        
        # 기본 기대 수익률 (의사결정 기반)
        base_return = 0.0
        if decision == "BUY":
            base_return = 0.05 * confidence  # 5% * 신뢰도
        elif decision == "SELL":
            base_return = -0.03 * confidence  # -3% * 신뢰도
        else:
            base_return = 0.0
        
        # 리스크 조정
        risk_adjusted_return = base_return * (1 - risk_score)
        
        return risk_adjusted_return

