"""
지식 증류 프레임워크
System 2 (Teacher) → System 1 (Student)
"""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class KnowledgeDistillation:
    """지식 증류 클래스"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        """지식 증류 초기화"""
        self.temperature = temperature  # 증류 온도
        self.alpha = alpha  # 하드 레이블 vs 소프트 레이블 가중치
    
    def distill_policy(self, 
                      teacher_policy: Dict[str, Any],
                      student_model: nn.Module,
                      student_input: torch.Tensor) -> Dict[str, Any]:
        """정책 증류"""
        # Teacher 정책을 소프트 레이블로 변환
        teacher_output = self._policy_to_tensor(teacher_policy)
        
        # Student 모델 출력
        student_output = student_model(student_input)
        
        # 증류 손실 계산
        loss = self._compute_distillation_loss(
            teacher_output=teacher_output,
            student_output=student_output,
            temperature=self.temperature
        )
        
        return {
            "loss": loss.item(),
            "teacher_output": teacher_output,
            "student_output": student_output
        }
    
    def distill_knowledge(self,
                         teacher_knowledge: Dict[str, Any],
                         student_model: nn.Module,
                         student_input: torch.Tensor) -> Dict[str, Any]:
        """지식 증류 (특징 증류)"""
        # Teacher의 중간 특징 추출 (실제로는 Teacher 모델 필요)
        # 여기서는 간단한 구현
        
        # Student 특징
        if hasattr(student_model, 'get_features'):
            student_features = student_model.get_features(student_input)
        else:
            # 기본: 마지막 레이어 전 특징
            student_features = student_input
        
        # 특징 매칭 손실 (간단한 구현)
        loss = nn.MSELoss()(
            student_features,
            torch.zeros_like(student_features)  # 실제로는 Teacher 특징 사용
        )
        
        return {
            "loss": loss.item(),
            "student_features": student_features
        }
    
    def distill_reasoning(self,
                         hypergraph_causal_paths: List[Dict[str, Any]],
                         student_model: nn.Module,
                         student_input: torch.Tensor) -> Dict[str, Any]:
        """추론 증류 (Reasoning Distillation) - 논문 핵심 기능"""
        # 1. 하이퍼그래프 인과 경로 임베딩 벡터 추출
        teacher_embeddings = self._extract_causal_path_embeddings(hypergraph_causal_paths)
        
        # 2. Student 모델의 중간 은닉층 벡터 추출
        if hasattr(student_model, 'get_features'):
            student_hidden = student_model.get_features(student_input)
        else:
            # 모델의 중간 레이어 추출 시도
            student_hidden = self._extract_hidden_features(student_model, student_input)
        
        # 3. 임베딩 벡터 차원 맞추기
        if teacher_embeddings.shape[-1] != student_hidden.shape[-1]:
            # 선형 변환으로 차원 맞추기
            if not hasattr(self, '_embedding_proj'):
                self._embedding_proj = nn.Linear(
                    teacher_embeddings.shape[-1],
                    student_hidden.shape[-1]
                ).to(student_hidden.device)
            teacher_embeddings = self._embedding_proj(teacher_embeddings)
        
        # 4. 특징 기반 증류 손실 계산 (MSE)
        reasoning_loss = nn.MSELoss()(student_hidden, teacher_embeddings)
        
        return {
            "loss": reasoning_loss.item(),
            "teacher_embeddings": teacher_embeddings,
            "student_hidden": student_hidden
        }
    
    def _extract_causal_path_embeddings(self, causal_paths: List[Dict[str, Any]]) -> torch.Tensor:
        """하이퍼그래프 인과 경로에서 임베딩 벡터 추출"""
        embeddings = []
        
        for path in causal_paths:
            # 경로의 노드 및 엣지 정보로부터 임베딩 생성
            nodes = path.get("nodes", [])
            edges = path.get("edges", [])
            weights = path.get("weights", [])
            te_scores = path.get("transfer_entropy", [])
            
            # 간단한 임베딩: 가중치와 전이 엔트로피 점수 기반
            if weights and te_scores:
                # 가중치와 TE 점수를 결합하여 임베딩 생성
                combined = np.array([w * te for w, te in zip(weights, te_scores)])
            elif weights:
                combined = np.array(weights)
            else:
                combined = np.array([0.5] * 10)  # 기본값
            
            # 정규화 및 패딩
            if len(combined) < 10:
                combined = np.pad(combined, (0, 10 - len(combined)), 'constant')
            elif len(combined) > 10:
                combined = combined[:10]
            
            embeddings.append(combined)
        
        if not embeddings:
            # 기본 임베딩 생성
            embeddings = [np.zeros(10)]
        
        return torch.tensor(np.array(embeddings), dtype=torch.float32)
    
    def _extract_hidden_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """모델의 중간 은닉층 특징 추출"""
        # 모델의 forward hook을 사용하여 중간 레이어 추출
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # 첫 번째 Linear 레이어에 hook 등록
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and len(features) == 0:
                handle = module.register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(x)
                handle.remove()
                break
        
        if features:
            # 배치 차원 처리
            hidden = features[0]
            if len(hidden.shape) == 3:  # [batch, seq, features]
                hidden = hidden[:, -1, :]  # 마지막 타임스텝
            elif len(hidden.shape) == 2:  # [batch, features]
                hidden = hidden
            else:
                hidden = hidden.flatten(1)
            
            return hidden
        else:
            # 기본값: 입력 데이터 사용
            if len(x.shape) == 3:
                return x[:, -1, :]
            return x
    
    def distill_value(self,
                     teacher_value: float,
                     student_model: nn.Module,
                     student_input: torch.Tensor) -> Dict[str, Any]:
        """가치 증류 (Value Distillation)"""
        # Student 모델의 가치 네트워크 출력
        if hasattr(student_model, 'value_head'):
            # 가치 헤드가 있는 경우
            with torch.no_grad():
                if hasattr(student_model, 'get_features'):
                    features = student_model.get_features(student_input)
                else:
                    features = student_input
                student_value = student_model.value_head(features)
        else:
            # 가치 헤드가 없는 경우: 의사결정 출력의 신뢰도로 근사
            student_output = student_model(student_input)
            if isinstance(student_output, tuple):
                logits, _ = student_output
            else:
                logits = student_output
            # 신뢰도를 가치로 근사
            probs = torch.softmax(logits, dim=-1)
            student_value = torch.max(probs, dim=-1)[0].unsqueeze(-1)
        
        # Teacher 가치를 텐서로 변환
        teacher_value_tensor = torch.tensor(
            teacher_value,
            dtype=torch.float32,
            device=student_value.device
        ).expand_as(student_value)
        
        # MSE 손실 계산
        value_loss = nn.MSELoss()(student_value, teacher_value_tensor)
        
        return {
            "loss": value_loss.item(),
            "teacher_value": teacher_value,
            "student_value": student_value.mean().item()
        }
    
    def _policy_to_tensor(self, policy: Dict[str, Any]) -> torch.Tensor:
        """정책을 텐서로 변환"""
        decision = policy.get("decision", "HOLD")
        confidence = policy.get("confidence", 0.5)
        
        # 결정을 원-핫 벡터로 변환 (논문 4.2.2: BUY, SELL, CLOSE, HOLD)
        decision_map = {"BUY": 0, "SELL": 1, "CLOSE": 2, "HOLD": 3}
        decision_idx = decision_map.get(decision, 3)
        
        # 소프트 레이블 생성 (논문 4.2.2: BUY, SELL, CLOSE, HOLD - 4차원)
        soft_label = torch.zeros(4)
        if decision_idx < 4:  # 유효한 인덱스인지 확인
            soft_label[decision_idx] = confidence
        # 나머지 확률 분산
        remaining = (1 - confidence) / 3  # 3개 행동에 분산
        for i in range(4):
            if i != decision_idx:
                soft_label[i] = remaining
        
        return soft_label
    
    def _compute_distillation_loss(self,
                                   teacher_output: torch.Tensor,
                                   student_output: torch.Tensor,
                                   temperature: float) -> torch.Tensor:
        """증류 손실 계산 (KL Divergence)"""
        # Temperature scaling
        teacher_soft = torch.softmax(teacher_output / temperature, dim=-1)
        student_soft = torch.softmax(student_output / temperature, dim=-1)
        
        # KL Divergence
        kl_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(student_soft + 1e-8),
            teacher_soft
        )
        
        return kl_loss * (temperature ** 2)
    
    def compute_total_loss(self,
                          task_loss: torch.Tensor,
                          policy_distillation_loss: Optional[torch.Tensor] = None,
                          reasoning_distillation_loss: Optional[torch.Tensor] = None,
                          value_distillation_loss: Optional[torch.Tensor] = None,
                          alpha: float = 0.6,
                          beta: float = 0.3,
                          gamma: float = 0.1,
                          volatility: float = 1.0) -> torch.Tensor:
        """전체 손실 계산 (논문 4.1.1)"""
        # 적응형 파라미터: 변동성에 따라 조정
        if volatility > 0.03:  # 높은 변동성 (위기 상황)
            alpha = min(alpha + 0.2, 0.9)  # 교사 지침 가중치 증가
        
        total_loss = (1 - alpha) * task_loss
        
        # Policy Distillation Loss
        if policy_distillation_loss is not None:
            total_loss += alpha * policy_distillation_loss
        
        # Reasoning Distillation Loss (논문 핵심)
        if reasoning_distillation_loss is not None:
            total_loss += beta * reasoning_distillation_loss
        
        # Value Distillation Loss
        if value_distillation_loss is not None:
            total_loss += gamma * value_distillation_loss
        
        return total_loss

