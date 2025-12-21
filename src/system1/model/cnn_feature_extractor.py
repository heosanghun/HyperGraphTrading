"""
1D-CNN 및 TCN 특징 추출기
논문 4.2.1: 틱 데이터 처리용 1D-CNN 또는 TCN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Conv1DFeatureExtractor(nn.Module):
    """1D-CNN 특징 추출기 (논문 4.2.1)"""
    
    def __init__(self,
                 input_dim: int = 10,
                 num_filters: int = 64,
                 kernel_sizes: list = [3, 5, 7],
                 dropout: float = 0.1):
        """1D-CNN 특징 추출기 초기화"""
        super(Conv1DFeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.num_filters = num_filters
        
        # 다중 커널 크기 컨볼루션 (Multi-scale Convolution)
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv)
        
        # 특징 결합
        self.combined_dim = num_filters * len(kernel_sizes)
        self.fc = nn.Sequential(
            nn.Linear(self.combined_dim, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        Args:
            x: [batch, seq_len, features] 또는 [batch, features, seq_len]
        Returns:
            extracted_features: [batch, num_filters]
        """
        # 입력 형태 확인 및 변환
        if len(x.shape) == 3:
            if x.shape[1] == self.input_dim:  # [batch, features, seq_len]
                x = x
            else:  # [batch, seq_len, features]
                x = x.transpose(1, 2)  # [batch, features, seq_len]
        else:
            # 2D 입력인 경우 차원 추가
            x = x.unsqueeze(-1)  # [batch, features, 1]
        
        # 다중 스케일 컨볼루션
        conv_outputs = []
        for conv in self.conv_layers:
            out = conv(x)  # [batch, num_filters, seq_len]
            # Global Average Pooling
            out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)  # [batch, num_filters]
            conv_outputs.append(out)
        
        # 특징 결합
        combined = torch.cat(conv_outputs, dim=1)  # [batch, combined_dim]
        
        # 최종 특징 추출
        features = self.fc(combined)  # [batch, num_filters]
        
        return features


class TCNBlock(nn.Module):
    """TCN (Temporal Convolutional Network) 블록"""
    
    def __init__(self,
                 num_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.1):
        """TCN 블록 초기화"""
        super(TCNBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        self.bn1 = nn.BatchNorm1d(num_channels)
        
        self.conv2 = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        self.bn2 = nn.BatchNorm1d(num_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # Causal padding을 위한 자르기
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Causal padding으로 인한 길이 차이 보정
        if out.shape[-1] != residual.shape[-1]:
            out = out[:, :, :residual.shape[-1]]
        
        out = self.relu(out + residual)
        return out


class TCNFeatureExtractor(nn.Module):
    """TCN 특징 추출기 (논문 4.2.1)"""
    
    def __init__(self,
                 input_dim: int = 10,
                 num_channels: int = 64,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        """TCN 특징 추출기 초기화"""
        super(TCNFeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.num_channels = num_channels
        
        # 입력 프로젝션
        self.input_proj = nn.Conv1d(input_dim, num_channels, 1)
        
        # TCN 블록들 (다이레이션 증가)
        self.tcn_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            block = TCNBlock(num_channels, kernel_size, dilation, dropout)
            self.tcn_blocks.append(block)
        
        # 출력 프로젝션
        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        Args:
            x: [batch, seq_len, features] 또는 [batch, features, seq_len]
        Returns:
            extracted_features: [batch, num_channels]
        """
        # 입력 형태 확인 및 변환
        if len(x.shape) == 3:
            if x.shape[1] == self.input_dim:  # [batch, features, seq_len]
                x = x
            else:  # [batch, seq_len, features]
                x = x.transpose(1, 2)  # [batch, features, seq_len]
        else:
            # 2D 입력인 경우 차원 추가
            x = x.unsqueeze(-1)  # [batch, features, 1]
        
        # 입력 프로젝션
        x = self.input_proj(x)  # [batch, num_channels, seq_len]
        
        # TCN 블록들 통과
        for block in self.tcn_blocks:
            x = block(x)
        
        # 출력 프로젝션
        features = self.output_proj(x)  # [batch, num_channels]
        
        return features

