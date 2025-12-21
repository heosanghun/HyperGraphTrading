"""
Saliency Map 추출 유틸리티
논문 4.1.1: System 1의 입력 데이터에 대한 Saliency Map 추출
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
import torch.nn.functional as F


class SaliencyMapExtractor:
    """Saliency Map 추출 클래스 (논문 4.1.1)"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """Saliency Map 추출기 초기화"""
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_gradient_based_saliency(self,
                                       input_tensor: torch.Tensor,
                                       target_class: Optional[int] = None) -> np.ndarray:
        """
        Gradient-based Saliency Map 추출
        Args:
            input_tensor: 입력 텐서 [batch, seq, features]
            target_class: 타겟 클래스 (None이면 예측된 클래스 사용)
        Returns:
            saliency_map: Saliency Map [seq, features]
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # 순전파
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        
        # 타겟 클래스 결정
        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()
        
        # 해당 클래스의 점수에 대한 그래디언트 계산
        score = logits[0, target_class]
        score.backward()
        
        # 입력에 대한 그래디언트 추출
        saliency = input_tensor.grad.abs().squeeze(0).cpu().numpy()  # [seq, features]
        
        return saliency
    
    def extract_integrated_gradients(self,
                                    input_tensor: torch.Tensor,
                                    baseline: Optional[torch.Tensor] = None,
                                    target_class: Optional[int] = None,
                                    steps: int = 50) -> np.ndarray:
        """
        Integrated Gradients Saliency Map 추출
        Args:
            input_tensor: 입력 텐서 [batch, seq, features]
            baseline: 베이스라인 텐서 (None이면 0 사용)
            target_class: 타겟 클래스
            steps: 적분 단계 수
        Returns:
            saliency_map: Saliency Map [seq, features]
        """
        input_tensor = input_tensor.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(self.device)
        
        # 타겟 클래스 결정
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output
                target_class = torch.argmax(logits, dim=-1).item()
        
        # 경로 적분
        alphas = torch.linspace(0, 1, steps).to(self.device)
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            output = self.model(interpolated)
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output
            
            score = logits[0, target_class]
            score.backward()
            
            integrated_grads += interpolated.grad
        
        # 평균 및 입력 차이 곱하기
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (input_tensor - baseline)
        
        saliency = integrated_grads.abs().squeeze(0).cpu().numpy()  # [seq, features]
        
        return saliency
    
    def extract_attention_saliency(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Attention-based Saliency Map 추출
        Args:
            input_tensor: 입력 텐서 [batch, seq, features]
        Returns:
            saliency_map: Saliency Map [seq, features]
        """
        input_tensor = input_tensor.to(self.device)
        
        # 모델의 어텐션 가중치 추출
        if hasattr(self.model, 'attention'):
            with torch.no_grad():
                # LSTM 출력
                if hasattr(self.model, 'lstm'):
                    lstm_out, _ = self.model.lstm(input_tensor)
                else:
                    lstm_out = input_tensor
                
                # 어텐션 적용
                attn_out, attn_weights = self.model.attention(
                    lstm_out, lstm_out, lstm_out
                )
                
                # 어텐션 가중치 평균
                if attn_weights is not None:
                    # [batch, num_heads, seq, seq] -> [seq, seq]
                    attn_weights = attn_weights.mean(dim=1).squeeze(0).cpu().numpy()
                    # 각 입력 위치의 중요도 (열 합계)
                    saliency = attn_weights.sum(axis=0)  # [seq]
                    # 특징 차원으로 확장
                    saliency = np.tile(saliency[:, np.newaxis], (1, input_tensor.shape[-1]))
                else:
                    saliency = np.ones((input_tensor.shape[1], input_tensor.shape[2]))
        else:
            # 어텐션 없으면 균등 분포
            saliency = np.ones((input_tensor.shape[1], input_tensor.shape[2]))
        
        return saliency
    
    def compare_with_system2_indicators(self,
                                       saliency_map: np.ndarray,
                                       system2_indicators: List[str],
                                       feature_names: List[str]) -> Dict[str, Any]:
        """
        System 2가 중요하게 여긴 거시 지표와 System 1의 가중치 일치 확인 (논문 4.1.1)
        Args:
            saliency_map: Saliency Map [seq, features]
            system2_indicators: System 2가 중요하게 여긴 지표 리스트
            feature_names: 특징 이름 리스트
        Returns:
            comparison_result: 비교 결과
        """
        # 특징별 평균 중요도
        feature_importance = saliency_map.mean(axis=0)  # [features]
        
        # System 2 지표와 매칭
        matches = []
        for indicator in system2_indicators:
            # 특징 이름에서 지표 찾기
            for i, name in enumerate(feature_names):
                if indicator.lower() in name.lower():
                    matches.append({
                        "indicator": indicator,
                        "feature_name": name,
                        "feature_index": i,
                        "importance": float(feature_importance[i])
                    })
                    break
        
        # 일치도 계산
        if matches:
            avg_importance = np.mean([m["importance"] for m in matches])
            match_score = avg_importance / (feature_importance.max() + 1e-8)
        else:
            match_score = 0.0
        
        return {
            "matches": matches,
            "match_score": float(match_score),
            "feature_importance": feature_importance.tolist(),
            "system2_indicators": system2_indicators
        }

