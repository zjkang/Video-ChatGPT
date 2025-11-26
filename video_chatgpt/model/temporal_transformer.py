import torch
import torch.nn as nn
from transformers.modeling_utils import apply_chunking_to_forward
import math


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2, num_heads: int = 8, max_seq_len: int = 100):
        """
        Temporal Transformer Encoder (T-LoRA 模块的核心结构)
        接收 [Batch, Time (100), Dim (1024)]
        输出 [Batch, Time, Dim (4096)]
        
        Args:
            input_dim: 输入特征维度（通常是视频特征维度，如 1024）
            output_dim: 输出特征维度（通常是 LLM 隐藏层维度，如 4096）
            num_layers: Transformer 层数（默认 2 层，保持轻量级）
            num_heads: 注意力头数（默认 8 头）
            max_seq_len: 最大序列长度（默认 100，对应 100 帧视频）
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # 1. 可学习的时间位置编码 (Learned Positional Encoding)
        # 用于帮助模型理解时间顺序
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # 2. 轻量级 Temporal Transformer Encoder Block
        # 这里是可训练的轻量级模块
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=input_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 最终投影层 (将 Transformer 输出维度投影到 LLM 隐藏层维度)
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            hidden_states: [Batch, Time, Input_Dim] 视频时序特征
            
        Returns:
            output_embeddings: [Batch, Time, Output_Dim] 时序建模后的特征
        """
        # hidden_states shape: [Batch, Time, Input_Dim]
        B, T, D = hidden_states.shape

        # 1. 加入位置编码
        if T > self.max_seq_len:
            # 如果序列长度超过最大值，使用插值或截断
            # 更好的做法是使用插值来适应更长的序列
            pos_embed = self._interpolate_pos_embed(T)
        else:
            pos_embed = self.temporal_pos_embed[:, :T, :]
            
        hidden_states = hidden_states + pos_embed

        # 2. 运行 Temporal Transformer
        transformer_output = self.transformer(hidden_states)

        # 3. 最终投影
        output_embeddings = self.output_proj(transformer_output)
        
        return output_embeddings
    
    def _interpolate_pos_embed(self, target_len: int) -> torch.Tensor:
        """
        当序列长度超过 max_seq_len 时，对位置编码进行插值
        
        Args:
            target_len: 目标序列长度
            
        Returns:
            插值后的位置编码 [1, target_len, input_dim]
        """
        # 使用 F.interpolate 进行 2D 插值（将时间维度视为空间维度）
        # 需要将 [1, max_seq_len, input_dim] 转换为 [1, input_dim, max_seq_len] 进行插值
        pos_embed = self.temporal_pos_embed.permute(0, 2, 1)  # [1, input_dim, max_seq_len]
        pos_embed = nn.functional.interpolate(
            pos_embed, 
            size=target_len, 
            mode='linear', 
            align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 1)  # [1, target_len, input_dim]
        return pos_embed

