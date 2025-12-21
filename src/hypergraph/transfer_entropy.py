"""
전이 엔트로피(Transfer Entropy) 기반 인과성 검증
논문 Appendix C.1 구현
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy import stats
from scipy.stats import entropy


def verify_causality(X: np.ndarray, 
                     Y: np.ndarray, 
                     theta: float = 2.0,
                     k: int = 1,
                     n_bins: int = 10) -> Tuple[bool, float, float]:
    """
    전이 엔트로피를 통한 인과성 검증
    
    Args:
        X: 원인 변수 시계열
        Y: 결과 변수 시계열
        theta: 유의성 임계값 (Z-score)
        k: 시차 (lag)
        n_bins: 히스토그램 빈 수
    
    Returns:
        (is_causal, te_score, z_score)
    """
    # 1. Stationarity Check (간단한 차분)
    X_stationary, Y_stationary = _ensure_stationarity(X, Y)
    
    # 2. Calculate Transfer Entropy
    te_score = calc_transfer_entropy(X_stationary, Y_stationary, k=k, n_bins=n_bins)
    
    # 3. Surrogate Data Test (Significance Test)
    surrogate_scores = []
    n_surrogates = 100
    
    for _ in range(n_surrogates):
        X_shuffled = np.random.permutation(X_stationary)
        te_surrogate = calc_transfer_entropy(X_shuffled, Y_stationary, k=k, n_bins=n_bins)
        surrogate_scores.append(te_surrogate)
    
    # 4. Z-Score Calculation
    if len(surrogate_scores) > 0 and np.std(surrogate_scores) > 0:
        z_score = (te_score - np.mean(surrogate_scores)) / np.std(surrogate_scores)
    else:
        z_score = 0.0
    
    # 5. Causal link 확인
    is_causal = z_score > theta
    
    return is_causal, te_score, z_score


def calc_transfer_entropy(X: np.ndarray, 
                         Y: np.ndarray, 
                         k: int = 1,
                         n_bins: int = 10) -> float:
    """
    전이 엔트로피 계산
    
    TE(X->Y) = H(Y_next | Y_curr) - H(Y_next | Y_curr, X_curr)
    
    Args:
        X: 원인 변수
        Y: 결과 변수
        k: 시차
        n_bins: 히스토그램 빈 수
    
    Returns:
        Transfer Entropy 값
    """
    if len(X) != len(Y) or len(X) < k + 2:
        return 0.0
    
    # 데이터 정규화 및 이산화
    X_norm = _normalize_and_bin(X, n_bins)
    Y_norm = _normalize_and_bin(Y, n_bins)
    
    # 조건부 엔트로피 계산
    # H(Y_next | Y_curr)
    h_y_next_given_y_curr = _conditional_entropy(
        Y_norm[k:],  # Y_next
        Y_norm[:-k] if k > 0 else Y_norm[:-1]  # Y_curr
    )
    
    # H(Y_next | Y_curr, X_curr)
    h_y_next_given_y_x = _conditional_entropy(
        Y_norm[k:],  # Y_next
        np.column_stack([
            Y_norm[:-k] if k > 0 else Y_norm[:-1],  # Y_curr
            X_norm[:-k] if k > 0 else X_norm[:-1]   # X_curr
        ])
    )
    
    # Transfer Entropy
    te = h_y_next_given_y_curr - h_y_next_given_y_x
    
    return max(0.0, te)  # 음수 방지


def _ensure_stationarity(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """정상성 확보 (간단한 차분)"""
    # ADF 테스트 대신 간단한 차분 사용
    X_diff = np.diff(X)
    Y_diff = np.diff(Y)
    
    # 길이 맞추기
    min_len = min(len(X_diff), len(Y_diff))
    return X_diff[:min_len], Y_diff[:min_len]


def _normalize_and_bin(data: np.ndarray, n_bins: int) -> np.ndarray:
    """데이터 정규화 및 이산화"""
    if len(data) == 0:
        return data
    
    # 정규화
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:
        normalized = (data - data_min) / (data_max - data_min)
    else:
        normalized = np.zeros_like(data)
    
    # 이산화
    binned = np.digitize(normalized, bins=np.linspace(0, 1, n_bins)) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    
    return binned.astype(int)


def _conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    """조건부 엔트로피 H(Y|X) 계산"""
    if len(y) != len(x):
        min_len = min(len(y), len(x))
        y = y[:min_len]
        x = x[:min_len]
    
    if len(y) == 0:
        return 0.0
    
    # 결합 분포 계산
    if x.ndim == 1:
        # 1D: 단순 조건부
        unique_x = np.unique(x)
        h_conditional = 0.0
        
        for x_val in unique_x:
            mask = x == x_val
            y_given_x = y[mask]
            
            if len(y_given_x) > 0:
                # P(X)
                p_x = np.sum(mask) / len(x)
                
                # H(Y|X=x)
                h_y_given_x = entropy(np.bincount(y_given_x) + 1e-10)
                
                h_conditional += p_x * h_y_given_x
        
        return h_conditional
    else:
        # 2D: 다변량 조건부
        # 간단한 구현: 각 조합에 대해 계산
        unique_combinations = {}
        for i in range(len(x)):
            x_tuple = tuple(x[i])
            if x_tuple not in unique_combinations:
                unique_combinations[x_tuple] = []
            unique_combinations[x_tuple].append(y[i])
        
        h_conditional = 0.0
        for x_tuple, y_values in unique_combinations.items():
            p_x = len(y_values) / len(y)
            h_y_given_x = entropy(np.bincount(y_values) + 1e-10)
            h_conditional += p_x * h_y_given_x
        
        return h_conditional


def verify_hyperedge_causality(hypergraph, 
                               node_ids: List[str],
                               theta: float = 2.0) -> Tuple[bool, float]:
    """
    하이퍼엣지의 인과성 검증
    
    Args:
        hypergraph: FinancialHypergraph 인스턴스
        node_ids: 하이퍼엣지에 포함된 노드 ID 리스트
        theta: 유의성 임계값
    
    Returns:
        (is_valid, avg_te_score)
    """
    if len(node_ids) < 2:
        return False, 0.0
    
    # 노드 데이터 추출
    node_data = {}
    for node_id in node_ids:
        node = hypergraph.get_node(node_id)
        if node:
            # 가격 데이터 추출
            if 'price_data' in node.features:
                data = np.array(node.features['price_data'])
            elif 'close' in node.features:
                data = np.array(node.features['close'])
            else:
                # 첫 번째 숫자 리스트 찾기
                data = None
                for key, value in node.features.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], (int, float)):
                            data = np.array(value)
                            break
                
                if data is None:
                    continue
            
            if len(data) > 10:  # 최소 데이터 포인트 필요
                node_data[node_id] = data
    
    if len(node_data) < 2:
        return False, 0.0
    
    # 모든 노드 쌍에 대해 전이 엔트로피 계산
    te_scores = []
    node_list = list(node_data.keys())
    
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            X = node_data[node_list[i]]
            Y = node_data[node_list[j]]
            
            # 길이 맞추기
            min_len = min(len(X), len(Y))
            X_aligned = X[:min_len]
            Y_aligned = Y[:min_len]
            
            # 양방향 검증
            is_causal_1, te_1, z_1 = verify_causality(X_aligned, Y_aligned, theta=theta)
            is_causal_2, te_2, z_2 = verify_causality(Y_aligned, X_aligned, theta=theta)
            
            # 더 강한 인과성 방향 선택
            if is_causal_1 or is_causal_2:
                te_scores.append(max(te_1, te_2))
    
    if len(te_scores) == 0:
        return False, 0.0
    
    avg_te = np.mean(te_scores)
    is_valid = len(te_scores) > 0 and avg_te > 0.1  # 최소 임계값
    
    return is_valid, avg_te

