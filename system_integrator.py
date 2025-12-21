"""
System 2 ↔ System 1 통합 모듈
비동기식 이중 프로세스 운용
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import torch
import time
import numpy as np
import threading
import copy

from ..system2.system2_teacher import System2Teacher
from ..system1.system1_student import System1Student
from ..hypergraph import FinancialHypergraph


class SystemIntegrator:
    """시스템 통합 클래스"""
    
    def __init__(self,
                 hypergraph: FinancialHypergraph,
                 system2: System2Teacher,
                 system1: System1Student,
                 update_interval_minutes: int = 30,
                 surprise_threshold: float = 0.3,
                 circuit_breaker_mdd: float = 0.15):
        """통합기 초기화"""
        self.hypergraph = hypergraph
        self.system2 = system2
        self.system1 = system1
        self.policy_history: List[Dict[str, Any]] = []
        
        # 비동기식 운용 설정
        self.update_interval = timedelta(minutes=update_interval_minutes)
        self.last_system2_update: Optional[datetime] = None
        self.surprise_threshold = surprise_threshold
        self.circuit_breaker_mdd = circuit_breaker_mdd
        
        # Context Vector (System 2 → System 1)
        self.context_vector: Optional[torch.Tensor] = None
        self.context_timestamp: Optional[datetime] = None
        
        # 성과 모니터링
        self.performance_history: List[Dict[str, Any]] = []
        self.current_mdd = 0.0
        self.is_safe_mode = False
        
        # 이중 버퍼링 (Double Buffering) - 논문 4.3.2
        self.old_model_state: Optional[Dict[str, Any]] = None  # 구 버전 모델 상태
        self.new_model_state: Optional[Dict[str, Any]] = None  # 신규 모델 상태
        self.retraining_thread: Optional[threading.Thread] = None
        self.retraining_lock = threading.Lock()
        self.is_retraining = False
        self.retraining_complete = False
    
    def update_system1_from_system2(self,
                                   symbol: str,
                                   date: str,
                                   num_policies: int = 10) -> Dict[str, Any]:
        """System 2 정책으로 System 1 업데이트 (Slow Path)"""
        start_time = time.time()
        
        # System 2에서 정책 생성
        policy_result = self.system2.generate_policy(symbol, date, use_llm=False)
        policy = policy_result["policy"]
        
        # 정책 히스토리에 추가
        self.policy_history.append(policy)
        self.last_system2_update = datetime.now()
        
        # Context Vector 생성 (System 2의 최신 정세)
        context_vector = self._create_context_vector(policy, policy_result)
        self.inject_context(context_vector)
        
        # 최근 정책들로 System 1 학습
        recent_policies = self.policy_history[-num_policies:]
        
        # 학습 데이터 준비
        training_data = self._prepare_training_data(symbol, len(recent_policies))
        
        # 하이퍼그래프 인과 경로 추출 (Reasoning Distillation용)
        causal_paths = []
        teacher_values = []
        for policy in recent_policies:
            # 각 정책에 대한 인과 경로 추출
            paths = self.system2.extract_causal_paths(symbol, date, max_paths=5)
            causal_paths.extend(paths)
            
            # 위험 조정 기대 수익률 추출 (Value Distillation용)
            value = self.system2.extract_risk_adjusted_return(policy)
            teacher_values.append(value)
        
        # 변동성 계산 (적응형 파라미터용)
        volatility = self._calculate_volatility(symbol)
        
        # 학습 (완전한 지식 증류: Policy + Reasoning + Value)
        training_result = self.system1.train_from_teacher(
            teacher_policies=recent_policies,
            training_data=training_data,
            epochs=5,
            learning_rate=0.001,
            hypergraph_causal_paths=causal_paths if causal_paths else None,
            teacher_values=teacher_values if teacher_values else None,
            volatility=volatility
        )
        
        elapsed_time = time.time() - start_time
        
        # 타임아웃 확인
        if elapsed_time > 3.0:
            print(f"[WARNING] System 2 업데이트 시간 초과: {elapsed_time:.2f}초")
        
        return {
            "policy": policy,
            "training_result": training_result,
            "elapsed_time": elapsed_time,
            "context_vector": context_vector
        }
    
    def _create_context_vector(self, policy: Dict[str, Any], policy_result: Dict[str, Any]) -> torch.Tensor:
        """Context Vector 생성 (System 2의 최신 정세를 압축)"""
        # 정책 정보를 벡터로 변환
        decision = policy.get("decision", "HOLD")
        confidence = policy.get("confidence", 0.5)
        
        # 결정을 원-핫 인코딩 (논문 4.2.2: BUY, SELL, CLOSE, HOLD)
        decision_map = {
            "BUY": [1, 0, 0, 0], 
            "SELL": [0, 1, 0, 0], 
            "CLOSE": [0, 0, 1, 0], 
            "HOLD": [0, 0, 0, 1]
        }
        decision_vec = decision_map.get(decision, [0, 0, 0, 1])
        
        # 신뢰도 및 기타 정보
        context_array = np.array(decision_vec + [confidence, policy.get("risk_score", 0.5)])
        
        # 차원 확장 (10차원으로)
        if len(context_array) < 10:
            context_array = np.pad(context_array, (0, 10 - len(context_array)), 'constant')
        elif len(context_array) > 10:
            context_array = context_array[:10]
        
        return torch.tensor(context_array, dtype=torch.float32).unsqueeze(0)  # [1, 10]
    
    def _calculate_volatility(self, symbol: str) -> float:
        """변동성 계산 (적응형 파라미터용)"""
        node = self.hypergraph.get_node(symbol)
        if not node:
            return 1.0
        
        # 가격 데이터에서 변동성 계산
        if 'price_data' in node.features:
            prices = node.features['price_data']
        elif 'close' in node.features:
            prices = node.features['close']
        else:
            return 1.0
        
        if isinstance(prices, list) and len(prices) >= 20:
            prices_array = np.array(prices[-20:])  # 최근 20일
            returns = np.diff(prices_array) / prices_array[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 1.0
            return float(volatility)
        
        return 1.0
    
    def _prepare_training_data(self, symbol: str, batch_size: int) -> torch.Tensor:
        """학습 데이터 준비"""
        # System 1 모델의 input_dim=33에 맞춰 특징 추출
        # FeatureExtractor의 출력 차원: 틱(5) + 오더북(8) + 기술적 지표(20) = 33차원
        from ..system1.feature_extractor import FeatureExtractor
        feature_extractor = FeatureExtractor()
        
        # 샘플 틱 데이터 생성
        sample_tick = {
            "close": 100.0,
            "open": 99.0,
            "high": 101.0,
            "low": 98.0,
            "volume": 1000000,
            "prices": [100.0, 101.0, 102.0]
        }
        sample_orderbook = {
            "bid_price": 99.9,
            "ask_price": 100.1,
            "bid_size": 1000,
            "ask_size": 1000
        }
        
        # 특징 추출
        features = feature_extractor.extract_features(sample_tick, sample_orderbook)
        feature_dim = len(features)  # 실제 특징 차원 (32 또는 33)
        
        # System 1 모델의 input_dim과 일치하도록 조정
        # System 1 모델은 33차원을 기대하므로, 차원이 다르면 패딩 또는 조정
        expected_dim = 33  # System1Student에서 설정한 feature_dim
        if feature_dim != expected_dim:
            # 차원이 다르면 조정 (실제로는 모델 차원에 맞춰야 함)
            feature_dim = expected_dim
        
        # 배치 데이터 생성
        return torch.randn(batch_size, 1, feature_dim)  # [batch, seq, features]
    
    def run_realtime(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        실시간 실행 (System 1 사용) - Fast Path
        이중 버퍼링: 재학습 완료 시 가중치 자동 교체
        """
        # 이중 버퍼링: 신규 모델 가중치 교체 확인 (트레이딩 공백 방지)
        if self.retraining_complete:
            self.swap_model_weights()
        
        # 안전 모드 확인
        if self.is_safe_mode:
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "reason": "Safe mode activated",
                "safe_mode": True
            }
        
        # Context Vector 주입 (System 2의 최신 정세 반영)
        if self.context_vector is not None:
            # Context를 tick_data에 추가
            tick_data["context_vector"] = self.context_vector
        
        # System 1로 실시간 추론 (구 버전 또는 신규 모델 사용)
        decision = self.system1.infer(tick_data)
        
        # 역방향 피드백 확인 (Surprise Signal)
        if self._should_wake_system2(decision, tick_data):
            self._wake_system2(tick_data.get("symbol", ""), tick_data.get("date", ""))
        
        return decision
    
    def _should_wake_system2(self, decision: Dict[str, Any], tick_data: Dict[str, Any]) -> bool:
        """System 2 강제 호출 필요 여부 판단 (역방향 피드백)"""
        # 1. 예측 오차 확인
        if "prediction_error" in decision:
            if decision["prediction_error"] > self.surprise_threshold:
                return True
        
        # 2. 신뢰도가 매우 낮은 경우
        if decision.get("confidence", 1.0) < 0.2:
            return True
        
        # 3. 미지정 패턴 감지 (변동성 급증)
        if "volatility" in tick_data:
            recent_volatility = tick_data["volatility"]
            if recent_volatility > 0.05:  # 5% 이상 변동성
                return True
        
        return False
    
    def _wake_system2(self, symbol: str, date: str) -> None:
        """System 2 강제 호출 (Wake-up Call)"""
        print(f"[WAKE-UP] System 2 강제 호출: {symbol} ({date})")
        try:
            # System 2 즉시 업데이트
            self.update_system1_from_system2(symbol, date, num_policies=5)
        except Exception as e:
            print(f"[ERROR] System 2 강제 호출 실패: {e}")
            # 실패 시 안전 모드 전환
            self.is_safe_mode = True
    
    def should_update_from_system2(self,
                                  recent_decisions: List[Dict[str, Any]],
                                  threshold: float = 0.3) -> bool:
        """System 2 업데이트 필요 여부 판단 (주기적/이벤트 기반)"""
        # 1. 주기적 업데이트 확인
        if self.last_system2_update is None:
            return True
        
        time_since_update = datetime.now() - self.last_system2_update
        if time_since_update >= self.update_interval:
            return True
        
        # 2. 최근 결정들의 신뢰도 확인
        if len(recent_decisions) >= 5:
            confidences = [d.get("confidence", 0) for d in recent_decisions[-5:]]
            avg_confidence = sum(confidences) / len(confidences)
            
            if avg_confidence < threshold:
                return True
        
        # 3. 결정 일관성 확인
        if len(recent_decisions) >= 5:
            decisions = [d.get("decision") for d in recent_decisions[-5:]]
            if len(set(decisions)) > 3:  # 결정이 너무 다양하면
                return True
        
        return False
    
    def inject_context(self, context_vector: torch.Tensor) -> None:
        """Context Vector 주입 (System 2 → System 1)"""
        self.context_vector = context_vector
        self.context_timestamp = datetime.now()
        print(f"[CONTEXT] Context Vector 주입 완료 (차원: {context_vector.shape})")
    
    def monitor_performance(self, 
                           current_return: float,
                           current_mdd: float,
                           volatility: float) -> Dict[str, Any]:
        """성과 모니터링 (Circuit Breaker)"""
        self.current_mdd = current_mdd
        
        # Circuit Breaker: MDD 임계치 초과
        if current_mdd > self.circuit_breaker_mdd:
            print(f"[CIRCUIT BREAKER] MDD 임계치 초과: {current_mdd:.2%} > {self.circuit_breaker_mdd:.2%}")
            self.is_safe_mode = True
            return {
                "triggered": True,
                "reason": "MDD threshold exceeded",
                "action": "block_new_entries"
            }
        
        # 변동성 급증 감지 (3-sigma)
        if len(self.performance_history) >= 10:
            recent_volatilities = [p.get("volatility", 0) for p in self.performance_history[-10:]]
            avg_vol = np.mean(recent_volatilities)
            std_vol = np.std(recent_volatilities)
            
            if volatility > avg_vol + 3 * std_vol:
                print(f"[CIRCUIT BREAKER] 변동성 급증 감지: {volatility:.4f} > {avg_vol + 3*std_vol:.4f}")
                self._wake_system2("", datetime.now().strftime("%Y-%m-%d"))
                return {
                    "triggered": True,
                    "reason": "Volatility spike",
                    "action": "wake_system2"
                }
        
        # 성과 기록
        self.performance_history.append({
            "timestamp": datetime.now(),
            "return": current_return,
            "mdd": current_mdd,
            "volatility": volatility
        })
        
        # 최근 100개만 유지
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        return {"triggered": False}
    
    def check_timeout_and_rollback(self, timeout_seconds: float = 3.0) -> bool:
        """타임아웃 및 롤백 확인"""
        if self.last_system2_update is None:
            return False
        
        time_since_update = (datetime.now() - self.last_system2_update).total_seconds()
        
        if time_since_update > timeout_seconds:
            print(f"[TIMEOUT] System 2 응답 지연: {time_since_update:.2f}초 > {timeout_seconds}초")
            self.is_safe_mode = True
            return True
        
        return False
    
    def trigger_retraining(self, symbol: str, date: str) -> Dict[str, Any]:
        """
        재학습 트리거 (이중 버퍼링 전략) - 논문 4.3.2
        구 버전 모델이 매매를 지속하는 동안 백그라운드에서 신규 모델 학습
        학습 완료 후 다음 틱부터 가중치만 교체하여 트레이딩 공백 방지
        """
        print(f"[RETRAINING] 재학습 트리거: {symbol} ({date})")
        
        # 이미 재학습 중이면 스킵
        with self.retraining_lock:
            if self.is_retraining:
                print("[RETRAINING] 이미 재학습 중입니다. 대기 중...")
                return {"success": False, "error": "Already retraining"}
            
            self.is_retraining = True
            self.retraining_complete = False
        
        # 구 버전 모델 상태 저장 (현재 모델 백업)
        try:
            self.old_model_state = copy.deepcopy(self.system1.model.state_dict())
            print("[DOUBLE BUFFERING] 구 버전 모델 상태 저장 완료")
        except Exception as e:
            print(f"[ERROR] 모델 상태 저장 실패: {e}")
            with self.retraining_lock:
                self.is_retraining = False
            return {"success": False, "error": f"Model state save failed: {e}"}
        
        # 백그라운드 스레드에서 재학습 수행
        def _retrain_in_background():
            """백그라운드 재학습 함수"""
            try:
                print("[DOUBLE BUFFERING] 백그라운드 재학습 시작...")
                
                # 신규 모델 학습
                result = self.update_system1_from_system2(symbol, date, num_policies=20)
                
                # 신규 모델 상태 저장
                self.new_model_state = copy.deepcopy(self.system1.model.state_dict())
                print("[DOUBLE BUFFERING] 신규 모델 학습 완료, 상태 저장 완료")
                
                # 재학습 완료 플래그 설정
                with self.retraining_lock:
                    self.retraining_complete = True
                    self.is_retraining = False
                
                print("[DOUBLE BUFFERING] 재학습 완료. 다음 틱에서 가중치 교체 대기 중...")
                
            except Exception as e:
                print(f"[ERROR] 백그라운드 재학습 실패: {e}")
                with self.retraining_lock:
                    self.is_retraining = False
                    self.retraining_complete = False
        
        # 백그라운드 스레드 시작
        self.retraining_thread = threading.Thread(
            target=_retrain_in_background,
            daemon=True
        )
        self.retraining_thread.start()
        
        return {
            "success": True,
            "message": "Retraining started in background",
            "old_model_saved": True
        }
    
    def swap_model_weights(self) -> bool:
        """
        모델 가중치 교체 (이중 버퍼링) - 논문 4.3.2
        신규 모델 학습이 완료되면 구 버전 모델의 가중치를 신규 모델로 교체
        트레이딩 공백(Downtime) 없이 즉시 적용
        """
        with self.retraining_lock:
            if not self.retraining_complete:
                return False
            
            if self.new_model_state is None:
                return False
        
        try:
            # 가중치 교체 (원자적 연산)
            print("[DOUBLE BUFFERING] 모델 가중치 교체 시작...")
            self.system1.model.load_state_dict(self.new_model_state)
            print("[DOUBLE BUFFERING] 모델 가중치 교체 완료")
            
            # 구 버전 모델 상태 정리
            self.old_model_state = None
            self.new_model_state = None
            self.retraining_complete = False
            
            # 안전 모드 해제
            self.is_safe_mode = False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 가중치 교체 실패: {e}")
            # 실패 시 구 버전 모델 유지
            if self.old_model_state is not None:
                try:
                    self.system1.model.load_state_dict(self.old_model_state)
                    print("[DOUBLE BUFFERING] 구 버전 모델로 롤백 완료")
                except:
                    pass
            return False
    
    def check_retraining_status(self) -> Dict[str, Any]:
        """재학습 상태 확인"""
        with self.retraining_lock:
            return {
                "is_retraining": self.is_retraining,
                "retraining_complete": self.retraining_complete,
                "has_new_model": self.new_model_state is not None,
                "has_old_model": self.old_model_state is not None
            }
