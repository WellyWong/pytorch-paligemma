# Based on Umar Jamil's implementation: https://github.com/hkproj/pytorch-paligemma
# Annotated with additional notes for educational purposes by Welly Wong
import torch
import torch.nn as nn
from typing import Optional, Tuple

class SiglipVisionConfig:
    """
    Configuration class for the Siglip Vision Transformer model.
    """
    def __init__(
        self,
        hidden_size=1152,          # size of the embedding vector (embed_dim) of the vision transformer
        intermediate_size=4304,    # up-projection size
        num_hidden_layers=27,      # the number of transformer encoder layer, stacked sequentially
        num_attention_heads=16,    # number of attention head in the multi head attention
        num_channels=3,            # the number of channels each image has (RGB)
        patch_size=14,             # kernel_size, the patches are taken with no overlap
        image_size=224,            # Paligemma has 3 different resolutions: 224x224, 448x448, and 896x896 pixels, we use the smallest resolution
        attention_dropout=0.0,
        layer_norm_eps=1e-6,
        num_image_tokens: int = 256,    # the number of image patches: (224/14)**2
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    """
    Converts input images into a sequence of patch embeddings for transformer models.
    Each image is split into non-overlapping patches, projected to `embed_dim`, and
    augmented with learnable positional embeddings.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(   # Conv2d acts like a patch extractor + linear projection in one
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",                # 'valid' means no padding
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2   # 256, num_patches is similar to seq_len in language model
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape            # [batch_size, channels, height, width]
        patch_embeds = self.patch_embedding(pixel_values)   # [batch_size, embed_dim, num_patches_h, num_patches_w]
        embeddings = patch_embeds.flatten(2)                # [batch_size, embed_dim, num_patches]
        embeddings = embeddings.transpose(1, 2)             # [batch_size, num_patches, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class SiglipAttention(nn.Module):
    """
    Implements multi-head self-attention for the Siglip model.
    Projects inputs into queries, keys, and values, applies scaled dot-product attention,
    and combines the results to produce context-aware embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5    # scaling factor 1/sqrt(dk)
        self.dropout = config.attention_dropout

        # The learned weights (W_q, W_k, W_v) make the same input behave differently,
        # model learns what kind of queries, keys, and values it needs via training
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)     # W_k
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)     # W_v
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)     # W_q
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)   # W_o

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        #    -> [batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # apply the softmax row-wise
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Take the weighted sum of all values, using the attn scores as weights, this is done simultaneously for each head
        attn_output = torch.matmul(attn_weights, value_states)   # [batch_size, num_heads, seq_len, head_dim]
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )            
        # [batch_size, seq_len, num_heads, head_dim], reorders dimensions to prepare for concatenating all heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # [batch_size, seq_len, embed_dim], can actually use .view() here since we already allocated contiguous memory
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # W_o (out_proj) learns to combine information across all heads
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

class SiglipMLP(nn.Module):
    """
    Feed-forward MLP block for the Siglip model.
    Projects inputs to a higher-dimensional intermediate space, applies GELU non-linearity,
    and projects back to the original embedding dimension.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")   # apply non-linearity
        
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    """
    Single transformer encoder layer for the Siglip model.
    Applies multi-head self-attention and a feed-forward MLP, each followed by layer normalization
    and residual connections to maintain the input shape [batch_size, num_patches, embed_dim].
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # The shape is maintained: [batch_size, num_patches, embed_dim]
        residual = hidden_states    # save this states for the skip connection
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states   # save this states for the 2nd skip connection
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states) 
        hidden_states = residual + hidden_states      
        return hidden_states

class SiglipEncoder(nn.Module):
    """
    Stack of transformer encoder layers for the Siglip model.
    Sequentially applies multiple SiglipEncoderLayer modules to input embeddings.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # The shape is maintained: [batch_size, num_patches, embed_dim]
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    """
    Vision transformer backbone for the Siglip model.
    Converts input images into patch embeddings, processes them through a stack of encoder layers,
    and applies final layer normalization to produce contextualized patch representations.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

class SiglipVisionModel(nn.Module):
    """
    Wrapper around the SiglipVisionTransformer.
    Takes images as input and returns the corresponding patch-level embeddings.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values) 