"""
실시간 추론 파이프라인
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

from ..model.architecture import LightweightTradingModel, SimplifiedTradingModel
from ..feature_extractor import FeatureExtractor
from ..utils.market_noise_filter import MarketNoiseFilter


class InferencePipeline:
    """추론 파이프라인"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 device: str = "cpu",
                 batch_size: int = 1,
                 use_orderbook: bool = True,
                 use_technical: bool = True):
        """파이프라인 초기화"""
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        
        # 특징 추출기 (논문 4.2.1)
        self.feature_extractor = FeatureExtractor(
            use_orderbook=use_orderbook,
            use_technical=use_technical
        )
        
        # 시장 소음 필터 (논문 4.2.2)
        self.noise_filter = MarketNoiseFilter()
        
        # 성능 측정
        self.inference_times = []
    
    def preprocess(self, tick_data: Dict[str, Any], orderbook_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """틱 데이터 전처리 (논문 4.2.1)"""
        # 통합 특징 추출기 사용
        features = self.feature_extractor.extract_features(tick_data, orderbook_data)
        
        # 정규화
        if np.max(np.abs(features)) > 0:
            features = features / (np.max(np.abs(features)) + 1e-8)
        
        # 시계열 형태로 변환 (1 timestep)
        features = features.reshape(1, 1, -1)  # [batch, seq, features]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def infer(self, data: torch.Tensor, 
              hyperedge_data: Optional[Dict[str, Any]] = None,
              market_volatility: Optional[float] = None) -> Dict[str, Any]:
        """추론 실행"""
        start_time = time.time()
        
        data = data.to(self.device)
        
        with torch.no_grad():
            if isinstance(self.model, (LightweightTradingModel, SimplifiedTradingModel)):
                output = self.model(data)
                if isinstance(output, tuple):
                    logits, value = output
                else:
                    logits = output
                    value = None
            else:
                logits = self.model(data)
                value = None
        
        # 시장 소음 필터 적용 (논문 4.2.2)
        hyperedge_strength = None
        transfer_entropy = None
        if hyperedge_data:
            hyperedge_strength = self.noise_filter.calculate_hyperedge_strength(hyperedge_data)
            transfer_entropy = self.noise_filter.get_transfer_entropy_from_hyperedge(hyperedge_data)
        
        filtered_logits = self.noise_filter.apply_noise_filter(
            logits,
            hyperedge_strength=hyperedge_strength,
            transfer_entropy=transfer_entropy,
            market_volatility=market_volatility
        )
        
        # 확률 계산
        probs = torch.softmax(filtered_logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        
        # 결정 매핑 (논문 4.2.2: BUY, SELL, CLOSE, HOLD)
        decision_map = {0: "BUY", 1: "SELL", 2: "CLOSE", 3: "HOLD"}
        decision = decision_map.get(prediction, "HOLD")
        confidence = probs[0, prediction].item()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # 확률 추출 (output_dim에 따라 동적 처리)
        num_actions = probs.shape[-1]
        probabilities = {}
        if num_actions >= 4:
            probabilities = {
                "BUY": probs[0, 0].item(),
                "SELL": probs[0, 1].item(),
                "CLOSE": probs[0, 2].item(),
                "HOLD": probs[0, 3].item()
            }
        else:
            # 하위 호환성 (기존 3개 행동)
            probabilities = {
                "BUY": probs[0, 0].item() if num_actions > 0 else 0.0,
                "SELL": probs[0, 1].item() if num_actions > 1 else 0.0,
                "CLOSE": 0.0,
                "HOLD": probs[0, 2].item() if num_actions > 2 else 0.0
            }
        
        result = {
            "decision": decision,
            "confidence": confidence,
            "probabilities": probabilities,
            "inference_time_ms": inference_time,
            "timestamp": datetime.now().isoformat(),
            "noise_filtered": hyperedge_strength is not None or transfer_entropy is not None
        }
        
        if value is not None:
            result["value"] = value.item()
        
        return result
    
    def postprocess(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """후처리"""
        # 신뢰도 기반 필터링
        if prediction["confidence"] < 0.5:
            prediction["decision"] = "HOLD"
            prediction["filtered"] = True
        else:
            prediction["filtered"] = False
        
        return prediction
    
    def execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """의사결정 실행 (시뮬레이션)"""
        execution = {
            "action": decision["decision"],
            "confidence": decision["confidence"],
            "timestamp": decision["timestamp"],
            "executed": True,
            "simulated": True  # 실제 거래는 별도 모듈에서
        }
        
        return execution
    
    def process_tick(self, tick_data: Dict[str, Any], 
                    orderbook_data: Optional[Dict[str, Any]] = None,
                    hyperedge_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """틱 데이터 처리 (전체 파이프라인)"""
        # 전처리
        processed_data = self.preprocess(tick_data, orderbook_data)
        
        # 시장 변동성 계산
        market_volatility = None
        if "prices" in tick_data and len(tick_data["prices"]) >= 20:
            prices = np.array(tick_data["prices"][-20:])
            returns = np.diff(prices) / prices[:-1]
            market_volatility = float(np.std(returns))
        
        # 추론
        prediction = self.infer(processed_data, hyperedge_data, market_volatility)
        
        # 후처리
        final_decision = self.postprocess(prediction)
        
        # 실행 (시뮬레이션)
        execution = self.execute_decision(final_decision)
        
        return {
            "prediction": final_decision,
            "execution": execution
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            "mean_inference_time_ms": float(np.mean(times)),
            "median_inference_time_ms": float(np.median(times)),
            "p95_inference_time_ms": float(np.percentile(times, 95)),
            "p99_inference_time_ms": float(np.percentile(times, 99)),
            "min_inference_time_ms": float(np.min(times)),
            "max_inference_time_ms": float(np.max(times)),
            "total_inferences": len(times)
        }

