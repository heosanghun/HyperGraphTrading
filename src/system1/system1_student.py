"""
System 1 (Student) 메인 클래스
경량 실시간 모델
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path

from .model.architecture import LightweightTradingModel, SimplifiedTradingModel
from .inference.pipeline import InferencePipeline
from .distillation.framework import KnowledgeDistillation


class System1Student:
    """System 1 (Student) 시스템"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_type: str = "simplified",
                 device: str = "cpu"):
        """System 1 초기화"""
        self.device = torch.device(device)
        self.model_type = model_type
        
        # 모델 생성 (논문 4.2.2: BUY, SELL, CLOSE, HOLD)
        # FeatureExtractor의 출력 차원에 맞춰 input_dim 설정
        # 기본 특징: 틱(5) + 오더북(8) + 기술적 지표(20) = 약 33차원
        # 간단히 사용하기 위해 33차원으로 설정 (실제로는 동적 계산 가능)
        feature_dim = 33  # FeatureExtractor의 예상 출력 차원
        
        if model_type == "lightweight":
            self.model = LightweightTradingModel(
                input_dim=feature_dim,
                hidden_dim=64,
                output_dim=4  # BUY, SELL, CLOSE, HOLD
            )
        else:  # simplified
            self.model = SimplifiedTradingModel(
                input_dim=feature_dim,
                hidden_dim=32,
                output_dim=4  # BUY, SELL, CLOSE, HOLD
            )
        
        self.model.to(self.device)
        
        # 모델 로드 (있는 경우)
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # 추론 파이프라인
        self.pipeline = InferencePipeline(
            model=self.model,
            device=device
        )
        
        # 지식 증류 (학습용)
        self.distillation = KnowledgeDistillation()
    
    def infer(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """실시간 추론"""
        return self.pipeline.process_tick(tick_data)
    
    def train_from_teacher(self,
                          teacher_policies: List[Dict[str, Any]],
                          training_data: torch.Tensor,
                          epochs: int = 10,
                          learning_rate: float = 0.001,
                          hypergraph_causal_paths: Optional[List[Dict[str, Any]]] = None,
                          teacher_values: Optional[List[float]] = None,
                          volatility: float = 1.0) -> Dict[str, Any]:
        """Teacher 정책으로부터 학습 (완전한 지식 증류)"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i, policy in enumerate(teacher_policies):
                if i >= len(training_data):
                    break
                
                # 입력 데이터
                x = training_data[i:i+1].to(self.device)
                
                # Teacher 정책을 타겟으로 변환 (논문 4.2.2: BUY, SELL, CLOSE, HOLD)
                decision = policy.get("decision", "HOLD")
                decision_map = {"BUY": 0, "SELL": 1, "CLOSE": 2, "HOLD": 3}
                target = torch.tensor([decision_map.get(decision, 3)], device=self.device)
                
                # 순전파
                output = self.model(x)
                if isinstance(output, tuple):
                    logits, value_output = output
                else:
                    logits = output
                    value_output = None
                
                # 1. Task Loss (기본 학습)
                task_loss = criterion(logits, target)
                
                # 2. Policy Distillation Loss
                policy_dist_loss_dict = self.distillation.distill_policy(
                    teacher_policy=policy,
                    student_model=self.model,
                    student_input=x
                )
                policy_dist_loss = torch.tensor(policy_dist_loss_dict["loss"], device=self.device)
                
                # 3. Reasoning Distillation Loss (논문 핵심)
                reasoning_dist_loss = None
                if hypergraph_causal_paths and i < len(hypergraph_causal_paths):
                    reasoning_dist_loss_dict = self.distillation.distill_reasoning(
                        hypergraph_causal_paths=[hypergraph_causal_paths[i]],
                        student_model=self.model,
                        student_input=x
                    )
                    reasoning_dist_loss = torch.tensor(reasoning_dist_loss_dict["loss"], device=self.device)
                
                # 4. Value Distillation Loss
                value_dist_loss = None
                if teacher_values and i < len(teacher_values) and value_output is not None:
                    value_dist_loss_dict = self.distillation.distill_value(
                        teacher_value=teacher_values[i],
                        student_model=self.model,
                        student_input=x
                    )
                    value_dist_loss = torch.tensor(value_dist_loss_dict["loss"], device=self.device)
                
                # 전체 손실 계산 (적응형 파라미터)
                total_loss = self.distillation.compute_total_loss(
                    task_loss=task_loss,
                    policy_distillation_loss=policy_dist_loss,
                    reasoning_distillation_loss=reasoning_dist_loss,
                    value_distillation_loss=value_dist_loss,
                    alpha=0.6,
                    beta=0.3,
                    gamma=0.1,
                    volatility=volatility
                )
                
                # 역전파
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(teacher_policies)
            training_losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        
        return {
            "training_losses": training_losses,
            "final_loss": training_losses[-1] if training_losses else 0.0
        }
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "model_config": self._get_model_config()
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
    
    def _get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        if isinstance(self.model, LightweightTradingModel):
            return {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "output_dim": self.model.output_dim
            }
        elif isinstance(self.model, SimplifiedTradingModel):
            return {
                "input_dim": 33,  # FeatureExtractor 출력 차원
                "hidden_dim": 32,
                "output_dim": 4  # BUY, SELL, CLOSE, HOLD
            }
        return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계"""
        return self.pipeline.get_performance_stats()

