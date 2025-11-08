import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import math


class DeepSeekV2RotaryEmbedding(nn.Module):
    """
    DeepSeek V2 Rotary Position Embedding with Yarn scaling support
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
        
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepSeekV2KVAccessor:
    """
    KV Cache accessor for DeepSeek V2 with support for different qk_head_dim and v_head_dim
    """
    
    def __init__(self, config, device=None, dtype=None):
        self.config = config
        self.device = device or torch.device('cuda')
        self.dtype = dtype or torch.float16
        
        # DeepSeek V2 specific dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        # DeepSeek V2 may have different dimensions for q/k and v
        self.qk_head_dim = getattr(config, 'qk_head_dim', self.head_dim)
        self.v_head_dim = getattr(config, 'v_head_dim', self.head_dim)
        
        # RoPE settings
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.rope_scaling = getattr(config, 'rope_scaling', None)
        
        # Initialize rotary embedding
        scaling_factor = 1.0
        if self.rope_scaling is not None:
            scaling_type = self.rope_scaling.get('type', 'linear')
            scaling_factor = self.rope_scaling.get('factor', 1.0)
            
        self.rotary_emb = DeepSeekV2RotaryEmbedding(
            self.qk_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            device=self.device,
            scaling_factor=scaling_factor
        )
        
    def create_empty_kv_cache(self, batch_size: int, max_length: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Create empty KV cache for all layers"""
        num_layers = self.config.num_hidden_layers
        
        keys = []
        values = []
        
        for _ in range(num_layers):
            # Key cache shape: (batch_size, num_key_value_heads, max_length, qk_head_dim)
            key_cache = torch.zeros(
                batch_size, self.num_key_value_heads, max_length, self.qk_head_dim,
                dtype=self.dtype, device=self.device
            )
            
            # Value cache shape: (batch_size, num_key_value_heads, max_length, v_head_dim)
            value_cache = torch.zeros(
                batch_size, self.num_key_value_heads, max_length, self.v_head_dim,
                dtype=self.dtype, device=self.device
            )
            
            keys.append(key_cache)
            values.append(value_cache)
            
        return keys, values
        
    def update_kv_cache_with_tree(
        self,
        past_key_values: Optional[Tuple],
        new_keys: List[torch.Tensor],
        new_values: List[torch.Tensor],
        tree_indices: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple:
        """Update KV cache with tree-structured new tokens
        
        Args:
            past_key_values: Existing KV cache
            new_keys: New key tensors for each layer
            new_values: New value tensors for each layer  
            tree_indices: Indices indicating tree structure for new tokens
            position_ids: Position IDs for RoPE
        """
        if past_key_values is None:
            # Initialize new cache
            updated_keys = new_keys
            updated_values = new_values
        else:
            # Extract existing keys and values
            old_keys = [kv[0] for kv in past_key_values]
            old_values = [kv[1] for kv in past_key_values]
            
            updated_keys = []
            updated_values = []
            
            for layer_idx, (old_k, old_v, new_k, new_v) in enumerate(
                zip(old_keys, old_values, new_keys, new_values)
            ):
                # Apply RoPE to new keys if position_ids provided
                if position_ids is not None:
                    cos, sin = self.rotary_emb(new_k, seq_len=position_ids.max().item() + 1)
                    # Select cos/sin for the new positions
                    new_cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
                    new_sin = sin[position_ids].unsqueeze(1)
                    
                    # Apply rotary embedding to new keys
                    new_k_rotated = (new_k * new_cos) + (rotate_half(new_k) * new_sin)
                else:
                    new_k_rotated = new_k
                    
                # Concatenate along sequence dimension
                updated_k = torch.cat([old_k, new_k_rotated], dim=2)
                updated_v = torch.cat([old_v, new_v], dim=2)
                
                updated_keys.append(updated_k)
                updated_values.append(updated_v)
                
        # Build past_key_values tuple
        return tuple((k, v) for k, v in zip(updated_keys, updated_values))
        
    def slice_kv_cache(
        self,
        past_key_values: Tuple,
        start_idx: int,
        end_idx: Optional[int] = None
    ) -> Tuple:
        """Slice KV cache along sequence dimension"""
        if past_key_values is None:
            return None
            
        sliced_kv = []
        for key, value in past_key_values:
            if end_idx is None:
                sliced_key = key[:, :, start_idx:]
                sliced_value = value[:, :, start_idx:]
            else:
                sliced_key = key[:, :, start_idx:end_idx]
                sliced_value = value[:, :, start_idx:end_idx]
                
            sliced_kv.append((sliced_key, sliced_value))
            
        return tuple(sliced_kv)
        
    def get_kv_seq_len(self, past_key_values: Optional[Tuple]) -> int:
        """Get sequence length from KV cache"""
        if past_key_values is None or len(past_key_values) == 0:
            return 0
            
        # Get sequence length from first layer's key cache
        first_key = past_key_values[0][0]
        return first_key.shape[2]
        
    def copy_kv_cache(
        self,
        past_key_values: Tuple,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor
    ) -> Tuple:
        """Copy KV cache entries from source to target indices"""
        if past_key_values is None:
            return None
            
        copied_kv = []
        for key, value in past_key_values:
            # Copy along sequence dimension
            copied_key = key.clone()
            copied_value = value.clone()
            
            # Copy source positions to target positions
            copied_key[:, :, target_indices] = key[:, :, source_indices]
            copied_value[:, :, target_indices] = value[:, :, source_indices]
            
            copied_kv.append((copied_key, copied_value))
            
        return tuple(copied_kv)
        
    def merge_kv_caches(
        self,
        cache_list: List[Tuple],
        merge_indices: List[torch.Tensor]
    ) -> Tuple:
        """Merge multiple KV caches according to merge indices"""
        if not cache_list:
            return None
            
        # Get dimensions from first cache
        first_cache = cache_list[0]
        num_layers = len(first_cache)
        
        merged_kv = []
        
        for layer_idx in range(num_layers):
            # Collect keys and values for this layer
            layer_keys = [cache[layer_idx][0] for cache in cache_list]
            layer_values = [cache[layer_idx][1] for cache in cache_list]
            
            # Merge along sequence dimension
            merged_key = torch.cat(layer_keys, dim=2)
            merged_value = torch.cat(layer_values, dim=2)
            
            merged_kv.append((merged_key, merged_value))
            
        return tuple(merged_kv)
        
    def apply_attention_mask_to_kv(
        self,
        past_key_values: Tuple,
        attention_mask: torch.Tensor
    ) -> Tuple:
        """Apply attention mask to KV cache (zero out masked positions)"""
        if past_key_values is None or attention_mask is None:
            return past_key_values
            
        masked_kv = []
        
        # attention_mask shape: [batch_size, seq_len]
        mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
        
        for key, value in past_key_values:
            # Apply mask to key and value
            masked_key = key * mask
            masked_value = value * mask
            
            masked_kv.append((masked_key, masked_value))
            
        return tuple(masked_kv)