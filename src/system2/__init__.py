"""
System 2 (Teacher) 모듈
하이퍼그래프 기반 근거 중심 토론 시스템
"""

from .system2_teacher import System2Teacher
from .agents.analyst_agent import AnalystAgent
from .agents.risk_agent import RiskAgent
from .agents.strategy_agent import StrategyAgent
from .discussion.framework import DiscussionFramework

__all__ = [
    'System2Teacher',
    'AnalystAgent',
    'RiskAgent',
    'StrategyAgent',
    'DiscussionFramework'
]
