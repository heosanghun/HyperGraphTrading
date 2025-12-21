"""
금융 하이퍼그래프 구축 모듈
"""
import networkx as nx
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import numpy as np
import pandas as pd
import uuid

from .structure import HyperNode, HyperEdge, NodeType, RelationType
from .transfer_entropy import verify_hyperedge_causality


class FinancialHypergraph:
    """금융 하이퍼그래프 클래스"""
    
    def __init__(self):
        """하이퍼그래프 초기화"""
        # 노드 저장 (id -> HyperNode)
        self.nodes: Dict[str, HyperNode] = {}
        
        # 하이퍼엣지 저장 (edge_id -> HyperEdge)
        self.edges: Dict[str, HyperEdge] = {}
        
        # 노드 타입별 인덱스
        self.node_index: Dict[NodeType, Set[str]] = {
            node_type: set() for node_type in NodeType
        }
        
        # NetworkX 그래프 (하이퍼엣지 시뮬레이션용)
        self.nx_graph = nx.Graph()
        
    def add_node(self, node: HyperNode) -> None:
        """노드 추가"""
        self.nodes[node.id] = node
        self.node_index[node.type].add(node.id)
        self.nx_graph.add_node(node.id, **node.features, type=node.type.value)
    
    def get_node(self, node_id: str) -> Optional[HyperNode]:
        """노드 조회"""
        return self.nodes.get(node_id)
    
    def add_hyperedge(self, edge: HyperEdge, verify_causality: bool = False, 
                     apply_regime_reweighting: bool = True) -> str:
        """하이퍼엣지 추가"""
        # 전이 엔트로피 검증 (선택적)
        if verify_causality:
            node_ids = edge.get_node_ids()
            is_valid, te_score = verify_hyperedge_causality(self, node_ids, theta=2.0)
            if not is_valid:
                # 검증 실패 시 신뢰도 낮춤
                edge.confidence = min(edge.confidence, 0.3)
            else:
                # 검증 성공 시 신뢰도 향상
                edge.confidence = min(edge.confidence + 0.2, 1.0)
                edge.evidence["transfer_entropy"] = te_score
        
        # 시장 국면 기반 가중치 조정 (선택적)
        if apply_regime_reweighting:
            from .analyzer import HypergraphAnalyzer
            analyzer = HypergraphAnalyzer(self)
            regime = analyzer.get_market_regime()
            if regime != "unknown":
                # 국면별 가중치 조정
                original_weight = edge.weight
                multiplier = self._get_regime_multiplier(regime, edge)
                edge.weight = min(1.0, max(0.0, original_weight * multiplier))
                edge.evidence["regime"] = regime
                edge.evidence["regime_multiplier"] = multiplier
        
        edge_id = self._generate_edge_id(edge)
        self.edges[edge_id] = edge
        
        # NetworkX 그래프에 클리크로 추가 (하이퍼엣지 시뮬레이션)
        node_ids = edge.get_node_ids()
        if len(node_ids) >= 2:
            # 완전 그래프로 연결 (하이퍼엣지 표현)
            for i, node1 in enumerate(node_ids):
                for node2 in node_ids[i+1:]:
                    self.nx_graph.add_edge(
                        node1, node2,
                        weight=edge.weight,
                        relation_type=edge.relation_type.value,
                        edge_id=edge_id,
                        confidence=edge.confidence
                    )
        
        return edge_id
    
    def _generate_edge_id(self, edge: HyperEdge) -> str:
        """엣지 ID 생성"""
        node_ids = sorted(edge.get_node_ids())
        return f"{edge.relation_type.value}_{'_'.join(node_ids)}"
    
    def compute_correlation(self, node_ids: List[str], method: str = "pearson") -> float:
        """노드 간 상관관계 계산"""
        if len(node_ids) < 2:
            return 0.0
        
        # 노드의 특징 데이터 추출
        features_list = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                # 다양한 특징 데이터 소스 시도
                if 'price_data' in node.features:
                    data = node.features['price_data']
                elif 'close' in node.features:
                    data = node.features['close']
                elif isinstance(node.features, dict) and len(node.features) > 0:
                    # 첫 번째 숫자 리스트 찾기
                    for key, value in node.features.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], (int, float)):
                                data = value
                                break
                    else:
                        continue
                else:
                    continue
                
                if isinstance(data, list) and len(data) > 1:
                    features_list.append(data)
        
        if len(features_list) < 2:
            return 0.0
        
        # 길이 맞추기
        min_len = min(len(f) for f in features_list)
        features_list = [f[:min_len] for f in features_list]
        
        # 상관관계 계산
        try:
            if method == "pearson":
                # 피어슨 상관계수
                corr_matrix = np.corrcoef(features_list)
                # 평균 상관계수 반환 (대각선 제외)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                if np.any(mask):
                    return float(np.mean(corr_matrix[mask]))
                else:
                    return 0.0
            else:
                return 0.0
        except Exception as e:
            return 0.0
    
    def update_dynamic(self, timestamp: datetime) -> None:
        """동적 업데이트 (시점별 그래프 업데이트)"""
        # 타임스탬프 기반 노드/엣지 필터링
        # 오래된 데이터 제거 또는 가중치 조정
        pass
    
    def get_evidence(self, query: str) -> List[Dict[str, Any]]:
        """하이퍼그래프에서 근거 추출"""
        evidence_list = []
        
        # 쿼리와 관련된 노드 찾기
        query_lower = query.lower()
        relevant_nodes = []
        
        for node_id, node in self.nodes.items():
            if query_lower in node_id.lower() or \
               any(query_lower in str(v).lower() for v in node.features.values()):
                relevant_nodes.append(node)
        
        # 관련 노드를 포함하는 하이퍼엣지 찾기
        for edge_id, edge in self.edges.items():
            for node in relevant_nodes:
                if edge.contains_node(node.id):
                    evidence_list.append({
                        "edge_id": edge_id,
                        "nodes": edge.get_node_ids(),
                        "relation": edge.relation_type.value,
                        "weight": edge.weight,
                        "confidence": edge.confidence,
                        "evidence": edge.evidence
                    })
                    break
        
        return evidence_list
    
    def get_neighbors(self, node_id: str, relation_type: Optional[RelationType] = None) -> List[str]:
        """노드의 이웃 노드 조회"""
        neighbors = set()
        
        for edge in self.edges.values():
            if edge.contains_node(node_id):
                for node in edge.nodes:
                    if node.id != node_id:
                        if relation_type is None or edge.relation_type == relation_type:
                            neighbors.add(node.id)
        
        return list(neighbors)
    
    def get_subgraph(self, node_ids: List[str]) -> 'FinancialHypergraph':
        """서브그래프 추출"""
        subgraph = FinancialHypergraph()
        
        # 노드 추가
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                subgraph.add_node(node)
        
        # 관련 엣지 추가
        for edge in self.edges.values():
            edge_node_ids = edge.get_node_ids()
            if all(nid in node_ids for nid in edge_node_ids):
                subgraph.add_hyperedge(edge)
        
        return subgraph
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (직렬화)"""
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "type": node.type.value,
                    "features": node.features,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                edge_id: {
                    "node_ids": edge.get_node_ids(),
                    "weight": edge.weight,
                    "relation_type": edge.relation_type.value,
                    "evidence": edge.evidence,
                    "confidence": edge.confidence
                }
                for edge_id, edge in self.edges.items()
            }
        }
    
    def create_event_node_from_news(self, news_item: Dict) -> HyperNode:
        """뉴스 데이터를 Event 노드로 변환 (1.3)"""
        
        # 노드 ID 생성
        news_id = f"news_{uuid.uuid4().hex[:8]}"
        if 'title' in news_item:
            # 제목 기반 간단한 ID
            title_hash = hash(str(news_item['title'])) % 1000000
            news_id = f"news_{abs(title_hash)}"
        
        # 타임스탬프
        timestamp = None
        if 'date' in news_item:
            if isinstance(news_item['date'], str):
                timestamp = pd.to_datetime(news_item['date'])
            elif isinstance(news_item['date'], datetime):
                timestamp = news_item['date']
        
        # 속성 구성
        features = {
            "title": str(news_item.get('title', '')),
            "text": str(news_item.get('text', '')),
            "source": str(news_item.get('source', '')),
            "url": str(news_item.get('url', '')),
            "sentiment": news_item.get('sentiment_polarity', 0.0),
            "sentiment_class": news_item.get('sentiment_class', 'neutral'),
            "urgency": news_item.get('urgency', 0.5),
            "event_type": news_item.get('event_type', 'General News'),
            "related_symbols": news_item.get('related_symbols', [])
        }
        
        # Event 노드 생성
        event_node = HyperNode(
            id=news_id,
            type=NodeType.NEWS,  # NEWS 타입 사용
            features=features,
            timestamp=timestamp
        )
        
        return event_node
    
    def add_news_to_hypergraph(self, news_data: pd.DataFrame) -> int:
        """뉴스 데이터를 하이퍼그래프에 통합 (1.4)"""
        added_count = 0
        
        if news_data.empty:
            return added_count
        
        print(f"[INFO] {len(news_data)}건의 뉴스 데이터를 하이퍼그래프에 추가 중...")
        
        for idx, row in news_data.iterrows():
            try:
                # 뉴스 → Event 노드 변환
                news_item = row.to_dict()
                event_node = self.create_event_node_from_news(news_item)
                
                # 노드 추가
                self.add_node(event_node)
                added_count += 1
                
                # 관련 주식과 연결 (하이퍼엣지 생성)
                related_symbols = news_item.get('related_symbols', [])
                if related_symbols:
                    self.create_hyperedge_from_news(event_node, related_symbols)
                
            except Exception as e:
                print(f"[WARNING] 뉴스 노드 추가 오류 (행 {idx}): {e}")
                continue
        
        print(f"[OK] {added_count}개의 뉴스 노드 추가 완료")
        return added_count
    
    def create_hyperedge_from_news(self, news_node: HyperNode, 
                                   related_assets: List[str]) -> Optional[HyperEdge]:
        """뉴스 이벤트와 관련 자산 간 하이퍼엣지 생성 (1.6)"""
        if not related_assets:
            return None
        
        # 관련 자산 노드 찾기
        asset_nodes = []
        for asset_id in related_assets:
            asset_node = self.get_node(asset_id)
            if asset_node:
                asset_nodes.append(asset_node)
        
        if not asset_nodes:
            return None
        
        # 하이퍼엣지 생성
        all_nodes = [news_node] + asset_nodes
        
        # 가중치 계산 (감성 및 긴급도 기반)
        sentiment = news_node.features.get('sentiment', 0.0)
        urgency = news_node.features.get('urgency', 0.5)
        weight = (abs(sentiment) * 0.6 + urgency * 0.4)
        weight = min(weight, 1.0)
        
        # 신뢰도 계산
        confidence = min(0.7 + urgency * 0.3, 1.0)
        
        hyperedge = HyperEdge(
            nodes=all_nodes,
            weight=weight,
            relation_type=RelationType.INFLUENCE,
            confidence=confidence,
            timestamp=news_node.timestamp,
            evidence={
                "source": "news_event",
                "event_type": news_node.features.get('event_type', 'General News'),
                "sentiment": sentiment,
                "urgency": urgency
            }
        )
        
        edge_id = self.add_hyperedge(hyperedge, apply_regime_reweighting=True)
        return hyperedge
    
    def _get_regime_multiplier(self, regime: str, edge: HyperEdge) -> float:
        """시장 국면에 따른 가중치 증폭 계수"""
        # 국면별 기본 증폭 계수
        if regime == "volatile":
            # 변동성 확대 시 리스크 엣지 증폭
            if edge.relation_type in [RelationType.INFLUENCE, RelationType.MARKET_IMPACT]:
                return 1.5
            elif edge.relation_type == RelationType.CORRELATION:
                return 1.3
            else:
                return 0.8
        elif regime == "bull":
            # 상승 시 긍정적 엣지 증폭
            if edge.weight > 0.5:
                return 1.2
            elif edge.relation_type in [RelationType.INFLUENCE, RelationType.MARKET_IMPACT]:
                return 0.9
            else:
                return 1.0
        elif regime == "bear":
            # 하락 시 리스크 엣지 증폭
            if edge.relation_type in [RelationType.INFLUENCE, RelationType.MARKET_IMPACT]:
                return 1.4
            elif edge.weight > 0.5:
                return 0.8
            else:
                return 1.0
        else:  # stable, normal
            return 1.0
    
    def create_macro_node(self, indicator: str, value: float, 
                         timestamp: datetime, 
                         trend: Optional[str] = None,
                         frequency: str = "Daily") -> HyperNode:
        """거시경제 지표를 Macro 노드로 변환 (2.2)"""
        node_id = f"macro_{indicator}"
        
        # 트렌드 계산 (없으면 자동 계산)
        if trend is None:
            # 기존 노드가 있으면 트렌드 계산
            existing_node = self.get_node(node_id)
            if existing_node and 'value' in existing_node.features:
                prev_value = existing_node.features.get('value', 0)
                if value > prev_value:
                    trend = "Upward"
                elif value < prev_value:
                    trend = "Downward"
                else:
                    trend = "Stable"
            else:
                trend = "Stable"
        
        features = {
            "indicator": indicator,
            "value": value,
            "trend": trend,
            "frequency": frequency
        }
        
        macro_node = HyperNode(
            id=node_id,
            type=NodeType.ECONOMIC,
            features=features,
            timestamp=timestamp
        )
        
        return macro_node
    
    def add_macro_to_hypergraph(self, macro_data: Dict[str, pd.DataFrame]) -> int:
        """거시경제 지표를 하이퍼그래프에 통합 (2.3)"""
        added_count = 0
        
        if not macro_data:
            return added_count
        
        print(f"[INFO] {len(macro_data)}개의 거시경제 지표를 하이퍼그래프에 추가 중...")
        
        for indicator, df in macro_data.items():
            if df.empty:
                continue
            
            try:
                # 각 날짜별로 노드 생성
                for idx, row in df.iterrows():
                    timestamp = None
                    if 'date' in row:
                        timestamp = pd.to_datetime(row['date'])
                    elif 'index' in row:
                        timestamp = pd.to_datetime(row['index'])
                    
                    # 가격 데이터 추출
                    value = None
                    if 'close' in row:
                        value = float(row['close'])
                    elif 'value' in row:
                        value = float(row['value'])
                    
                    if value is None or pd.isna(value):
                        continue
                    
                    # Macro 노드 생성
                    macro_node = self.create_macro_node(
                        indicator=indicator,
                        value=value,
                        timestamp=timestamp,
                        frequency="Daily"
                    )
                    
                    # 노드 추가 또는 업데이트
                    existing_node = self.get_node(macro_node.id)
                    if existing_node:
                        # 기존 노드 업데이트
                        existing_node.features.update(macro_node.features)
                        existing_node.timestamp = timestamp
                    else:
                        # 새 노드 추가
                        self.add_node(macro_node)
                        added_count += 1
                
                # 관련 주식과 연결 (하이퍼엣지 생성)
                affected_assets = self._get_affected_assets_by_macro(indicator)
                if affected_assets:
                    latest_node = self.get_node(f"macro_{indicator}")
                    if latest_node:
                        self.create_hyperedge_from_macro(latest_node, affected_assets)
                
            except Exception as e:
                print(f"[WARNING] 거시경제 지표 추가 오류 ({indicator}): {e}")
                continue
        
        print(f"[OK] {added_count}개의 거시경제 노드 추가 완료")
        return added_count
    
    def _get_affected_assets_by_macro(self, indicator: str) -> List[str]:
        """거시경제 지표에 영향받는 자산 리스트 반환"""
        # 간단한 규칙 기반 매핑
        macro_asset_mapping = {
            "US10Y": ["JPM", "BAC", "WFC", "C"],  # 금리 상승 → 금융주
            "US2Y": ["JPM", "BAC", "WFC", "C"],
            "DXY": ["AAPL", "MSFT", "GOOGL"],  # 달러 강세 → 수출주
            "VIX": ["SPY", "QQQ"],  # 변동성 → 지수
            "WTI": ["XOM", "CVX", "SLB"],  # 유가 → 에너지주
            "GOLD": ["GLD", "GDX"],  # 금 → 금 관련 ETF
        }
        
        return macro_asset_mapping.get(indicator, [])
    
    def create_hyperedge_from_macro(self, macro_node: HyperNode,
                                   affected_assets: List[str]) -> Optional[HyperEdge]:
        """거시경제 지표와 영향받는 자산 간 하이퍼엣지 생성 (2.4)"""
        if not affected_assets:
            return None
        
        # 영향받는 자산 노드 찾기
        asset_nodes = []
        for asset_id in affected_assets:
            asset_node = self.get_node(asset_id)
            if asset_node:
                asset_nodes.append(asset_node)
        
        if not asset_nodes:
            return None
        
        # 하이퍼엣지 생성
        all_nodes = [macro_node] + asset_nodes
        
        # 가중치 계산 (트렌드 기반)
        trend = macro_node.features.get('trend', 'Stable')
        if trend == "Upward":
            weight = 0.7
        elif trend == "Downward":
            weight = 0.7
        else:
            weight = 0.5
        
        # 신뢰도 계산
        confidence = 0.8  # 거시경제 지표는 높은 신뢰도
        
        hyperedge = HyperEdge(
            nodes=all_nodes,
            weight=weight,
            relation_type=RelationType.MARKET_IMPACT,
            confidence=confidence,
            timestamp=macro_node.timestamp,
            evidence={
                "source": "macro_indicator",
                "indicator": macro_node.features.get('indicator', ''),
                "value": macro_node.features.get('value', 0),
                "trend": trend
            }
        )
        
        edge_id = self.add_hyperedge(hyperedge)
        return hyperedge
    
    def create_option_node(self, option_data: Dict) -> HyperNode:
        """옵션 데이터를 Option 노드로 변환 (논문 5.1.1)"""
        option_id = f"option_{option_data.get('underlying', 'UNKNOWN')}_{option_data.get('strike', 0)}_{option_data.get('expiration', '')}"
        
        features = {
            "underlying": option_data.get('underlying', ''),
            "strike": option_data.get('strike', 0.0),
            "expiration": str(option_data.get('expiration', '')),
            "option_type": option_data.get('option_type', 'call'),
            "option_price": option_data.get('option_price', 0.0),
            "implied_volatility": option_data.get('implied_volatility', 0.0),
            "delta": option_data.get('delta', 0.0),
            "gamma": option_data.get('gamma', 0.0),
            "theta": option_data.get('theta', 0.0),
            "vega": option_data.get('vega', 0.0),
            "bid": option_data.get('bid', 0.0),
            "ask": option_data.get('ask', 0.0),
            "volume": option_data.get('volume', 0),
            "open_interest": option_data.get('openInterest', 0)
        }
        
        timestamp = None
        if 'expiration' in option_data:
            try:
                timestamp = pd.to_datetime(option_data['expiration'])
            except:
                pass
        
        option_node = HyperNode(
            id=option_id,
            type=NodeType.OPTION,
            features=features,
            timestamp=timestamp
        )
        
        return option_node
    
    def create_futures_node(self, futures_data: Dict) -> HyperNode:
        """선물 데이터를 Futures 노드로 변환 (논문 5.1.1)"""
        futures_id = f"futures_{futures_data.get('futures_symbol', 'UNKNOWN')}"
        
        features = {
            "symbol": futures_data.get('futures_symbol', ''),
            "close": futures_data.get('close', 0.0),
            "open": futures_data.get('open', 0.0),
            "high": futures_data.get('high', 0.0),
            "low": futures_data.get('low', 0.0),
            "volume": futures_data.get('volume', 0),
            "instrument_type": "futures"
        }
        
        timestamp = None
        if 'date' in futures_data:
            try:
                timestamp = pd.to_datetime(futures_data['date'])
            except:
                pass
        
        futures_node = HyperNode(
            id=futures_id,
            type=NodeType.FUTURES,
            features=features,
            timestamp=timestamp
        )
        
        return futures_node
    
    def add_option_to_hypergraph(self, option_data: pd.DataFrame) -> int:
        """옵션 데이터를 하이퍼그래프에 통합 (논문 5.1.1)"""
        added_count = 0
        
        if option_data.empty:
            return added_count
        
        print(f"[INFO] {len(option_data)}건의 옵션 데이터를 하이퍼그래프에 추가 중...")
        
        for idx, row in option_data.iterrows():
            try:
                option_dict = row.to_dict()
                option_node = self.create_option_node(option_dict)
                
                # 노드 추가
                if option_node.id not in self.nodes:
                    self.add_node(option_node)
                    added_count += 1
                
                # 기초자산과의 하이퍼엣지 생성
                underlying = option_dict.get('underlying', '')
                if underlying and self.get_node(underlying):
                    hyperedge = self.create_hyperedge_from_option(option_node, underlying)
                    if hyperedge:
                        self.add_hyperedge(hyperedge)
                
            except Exception as e:
                print(f"  [ERROR] 옵션 데이터 추가 오류 (인덱스 {idx}): {e}")
                continue
        
        print(f"  [OK] {added_count}개의 옵션 노드 추가 완료")
        return added_count
    
    def create_hyperedge_from_option(self, option_node: HyperNode, underlying_symbol: str) -> Optional[HyperEdge]:
        """옵션과 기초자산 간 하이퍼엣지 생성"""
        underlying_node = self.get_node(underlying_symbol)
        if not underlying_node:
            return None
        
        # 옵션의 Delta를 가중치로 사용 (기초자산 가격 변화에 대한 민감도)
        delta = abs(option_node.features.get('delta', 0.5))
        weight = min(delta, 1.0)
        
        hyperedge = HyperEdge(
            nodes=[option_node, underlying_node],
            weight=weight,
            relation_type=RelationType.INFLUENCE,
            evidence={
                "source": "option_delta",
                "delta": delta,
                "iv": option_node.features.get('implied_volatility', 0.0),
                "option_type": option_node.features.get('option_type', 'call')
            },
            confidence=delta
        )
        
        return hyperedge
    
    def add_futures_to_hypergraph(self, futures_data: pd.DataFrame) -> int:
        """선물 데이터를 하이퍼그래프에 통합 (논문 5.1.1)"""
        added_count = 0
        
        if futures_data.empty:
            return added_count
        
        print(f"[INFO] {len(futures_data)}건의 선물 데이터를 하이퍼그래프에 추가 중...")
        
        for idx, row in futures_data.iterrows():
            try:
                futures_dict = row.to_dict()
                futures_node = self.create_futures_node(futures_dict)
                
                # 노드 추가
                if futures_node.id not in self.nodes:
                    self.add_node(futures_node)
                    added_count += 1
                
            except Exception as e:
                print(f"  [ERROR] 선물 데이터 추가 오류 (인덱스 {idx}): {e}")
                continue
        
        print(f"  [OK] {added_count}개의 선물 노드 추가 완료")
        return added_count

