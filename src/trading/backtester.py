"""
백테스팅 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class Backtester:
    """백테스터"""
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001):
        """백테스터 초기화"""
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self) -> None:
        """리셋"""
        self.cash = self.initial_capital
        self.position = 0.0
        self.shares = 0
        self.equity = []
        self.trades = []
        self.current_price = 0.0
    
    def execute_trade(self,
                     decision: str,
                     price: float,
                     confidence: float,
                     timestamp: str) -> Dict[str, Any]:
        """거래 실행"""
        trade = {
            "timestamp": timestamp,
            "decision": decision,
            "price": price,
            "confidence": confidence,
            "shares": 0,
            "cost": 0.0,
            "executed": False
        }
        
        if decision == "BUY" and self.cash > 0:
            # 매수
            max_shares = int(self.cash / (price * (1 + self.transaction_cost)))
            if max_shares > 0:
                shares_to_buy = int(max_shares * confidence)  # 신뢰도 기반 포지션 크기
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                
                if cost <= self.cash:
                    self.shares += shares_to_buy
                    self.cash -= cost
                    trade["shares"] = shares_to_buy
                    trade["cost"] = cost
                    trade["executed"] = True
        
        elif decision == "SELL" and self.shares > 0:
            # 매도
            shares_to_sell = int(self.shares * confidence)
            if shares_to_sell > 0:
                revenue = shares_to_sell * price * (1 - self.transaction_cost)
                self.shares -= shares_to_sell
                self.cash += revenue
                trade["shares"] = -shares_to_sell
                trade["cost"] = revenue
                trade["executed"] = True
        
        if trade["executed"]:
            self.trades.append(trade)
        
        self.current_price = price
        return trade
    
    def update_equity(self, current_price: float) -> None:
        """자산 가치 업데이트"""
        portfolio_value = self.cash + self.shares * current_price
        self.equity.append({
            "cash": self.cash,
            "shares": self.shares,
            "share_value": self.shares * current_price,
            "total_value": portfolio_value
        })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """성능 지표 계산"""
        if not self.equity:
            return {}
        
        equity_values = [e["total_value"] for e in self.equity]
        
        # 수익률
        total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital
        
        # 일일 수익률
        daily_returns = []
        for i in range(1, len(equity_values)):
            daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
            daily_returns.append(daily_return)
        
        if not daily_returns:
            return {"total_return": total_return}
        
        # Sharpe Ratio
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        sharpe_ratio = (mean_return / (std_return + 1e-8)) * np.sqrt(252) if std_return > 0 else 0.0
        
        # Maximum Drawdown
        peak = equity_values[0]
        max_drawdown = 0.0
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 승률
        winning_trades = sum(1 for t in self.trades if t.get("profit", 0) > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": len(self.trades),
            "final_value": equity_values[-1],
            "mean_daily_return": mean_return,
            "volatility": std_return
        }
    
    def run_backtest(self,
                    price_data: pd.DataFrame,
                    decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """백테스팅 실행"""
        self.reset()
        
        # 가격 데이터와 결정 매칭
        for i, decision in enumerate(decisions):
            if i >= len(price_data):
                break
            
            row = price_data.iloc[i]
            price = row.get("close", row.get("Close", 0))
            timestamp = row.get("timestamp", row.get("Date", ""))
            
            if price > 0:
                # 거래 실행
                self.execute_trade(
                    decision=decision.get("decision", "HOLD"),
                    price=price,
                    confidence=decision.get("confidence", 0.5),
                    timestamp=str(timestamp)
                )
                
                # 자산 가치 업데이트
                self.update_equity(price)
        
        # 최종 정리 (보유 주식 매도)
        if self.shares > 0 and len(price_data) > 0:
            final_price = price_data.iloc[-1].get("close", price_data.iloc[-1].get("Close", 0))
            if final_price > 0:
                revenue = self.shares * final_price * (1 - self.transaction_cost)
                self.cash += revenue
                self.shares = 0
        
        # 성능 지표 계산
        metrics = self.calculate_metrics()
        
        return {
            "metrics": metrics,
            "trades": self.trades,
            "equity_curve": self.equity,
            "final_cash": self.cash,
            "final_shares": self.shares
        }

