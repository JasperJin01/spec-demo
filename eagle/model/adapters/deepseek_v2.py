import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from transformers import DeepseekV2ForCausalLM, DeepseekV2Config


class DeepseekV2HFAdapter:
    """
    Adapter for DeepSeek V2 HuggingFace model to provide unified interface for EAGLE3.
    Handles device management, lm_head access, and KV cache operations.
    """
    
    def __init__(self, base_model: DeepseekV2ForCausalLM):
        self.base_model = base_model
        self.config = base_model.config
        self.device = next(base_model.parameters()).device
        self.dtype = next(base_model.parameters()).dtype
        
        # Cache model components for efficient access
        self.model = base_model.model
        self.lm_head = base_model.lm_head
        self.embed_tokens = self.model.embed_tokens
        
        # DeepSeek V2 specific parameters
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.config.hidden_size // self.num_heads
        
        # DeepSeek V2 may have different dimensions for q/k and v
        self.qk_head_dim = getattr(self.config, 'qk_head_dim', self.head_dim)
        self.v_head_dim = getattr(self.config, 'v_head_dim', self.head_dim)
        
    def get_device(self) -> torch.device:
        """Get model device"""
        return self.device
        
    def get_dtype(self) -> torch.dtype:
        """Get model dtype"""
        return self.dtype
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.config.vocab_size
        
    def get_hidden_size(self) -> int:
        """Get hidden size"""
        return self.config.hidden_size
        
    def get_embed_tokens(self) -> nn.Embedding:
        """Get embedding layer"""
        return self.embed_tokens
        
    def get_lm_head(self) -> nn.Linear:
        """Get language model head"""
        return self.lm_head
        
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states using lm_head"""
        return self.lm_head(hidden_states)
        
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for generation, similar to HF's prepare_inputs_for_generation"""
        
        # Handle position_ids
        position_ids = kwargs.get('position_ids', None)
        if position_ids is None:
            if past_key_values is not None:
                # Get sequence length from past_key_values
                past_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
                position_ids = torch.arange(
                    past_length, past_length + input_ids.shape[1],
                    dtype=torch.long, device=input_ids.device
                ).unsqueeze(0)
            else:
                position_ids = torch.arange(
                    input_ids.shape[1], dtype=torch.long, device=input_ids.device
                ).unsqueeze(0)
                
        # Handle attention_mask
        if attention_mask is None:
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
                attention_mask = torch.ones(
                    (input_ids.shape[0], past_length + input_ids.shape[1]),
                    dtype=torch.bool, device=input_ids.device
                )
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': use_cache,
        }
        
    def forward_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass with KV cache support"""
        
        inputs = self.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )
        
        outputs = self.base_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            position_ids=inputs['position_ids'],
            past_key_values=inputs['past_key_values'],
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        return {
            'logits': outputs.logits,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'last_hidden_state': outputs.hidden_states[-1] if output_hidden_states else None
        }
        
    def get_kv_cache_shape(self, batch_size: int, max_length: int) -> Tuple[int, ...]:
        """Get the shape for KV cache tensors"""
        # DeepSeek V2 KV cache shape: (batch_size, num_key_value_heads, seq_len, head_dim)
        return (batch_size, self.num_key_value_heads, max_length, self.qk_head_dim)
        
    def get_v_cache_shape(self, batch_size: int, max_length: int) -> Tuple[int, ...]:
        """Get the shape for V cache tensors (may differ from K cache in DeepSeek V2)"""
        return (batch_size, self.num_key_value_heads, max_length, self.v_head_dim)
        
    def extract_kv_from_past(self, past_key_values: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract key and value tensors from past_key_values tuple"""
        if past_key_values is None or len(past_key_values) == 0:
            return None, None
            
        # past_key_values is a tuple of (key, value) pairs for each layer
        keys = []
        values = []
        
        for layer_past in past_key_values:
            if layer_past is not None:
                key, value = layer_past
                keys.append(key)
                values.append(value)
            else:
                keys.append(None)
                values.append(None)
                
        return keys, values
        
    def build_past_key_values(self, keys: list, values: list) -> Tuple:
        """Build past_key_values tuple from key and value lists"""
        if not keys or not values:
            return None
            
        past_key_values = []
        for key, value in zip(keys, values):
            if key is not None and value is not None:
                past_key_values.append((key, value))
            else:
                past_key_values.append(None)
                
        return tuple(past_key_values)
        
    def update_kv_cache(
        self,
        past_key_values: Optional[Tuple],
        new_keys: list,
        new_values: list,
        cache_position: Optional[torch.Tensor] = None
    ) -> Tuple:
        """Update KV cache with new key-value pairs"""
        if past_key_values is None:
            return self.build_past_key_values(new_keys, new_values)
            
        old_keys, old_values = self.extract_kv_from_past(past_key_values)
        updated_keys = []
        updated_values = []
        
        for i, (old_k, old_v, new_k, new_v) in enumerate(zip(old_keys, old_values, new_keys, new_values)):
            if old_k is not None and new_k is not None:
                # Concatenate along sequence dimension (dim=2)
                updated_k = torch.cat([old_k, new_k], dim=2)
                updated_v = torch.cat([old_v, new_v], dim=2)
            elif new_k is not None:
                updated_k = new_k
                updated_v = new_v
            else:
                updated_k = old_k
                updated_v = old_v
                
            updated_keys.append(updated_k)
            updated_values.append(updated_v)
            
        return self.build_past_key_values(updated_keys, updated_values)