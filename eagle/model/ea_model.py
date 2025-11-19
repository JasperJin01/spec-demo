import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer as _AutoTokenizerForDraft

class ExternalDraftAdapter:
    """
    外部草稿模型封装：管理小模型、稳定 KV 缓存，以及递归树展开。
    不依赖主模型隐藏层，直接用小模型 logits 构建草稿树。
    """
    def __init__(self, small_model, tokenizer, device, dtype):
        self.model = small_model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.stable_kv = None
        self.model.to(self.device)
        self.model.eval()

    def reset_kv(self):
        self.stable_kv = None

    @torch.no_grad()
    def _advance_prefix(self, context_ids):
        # 将前缀输入写入 stable_kv
        if self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            new_prefix = context_ids[:, kv_len:]
            if new_prefix.shape[1] > 0:
                out_ctx = self.model(input_ids=new_prefix, use_cache=True, past_key_values=self.stable_kv)
                self.stable_kv = out_ctx.past_key_values
        else:
            if context_ids.shape[1] > 0:
                out_ctx = self.model(input_ids=context_ids, use_cache=True)
                self.stable_kv = out_ctx.past_key_values

    @torch.no_grad()
    def propose_tree(self, input_ids, total_tokens, depth, top_k, tree_mask_init, position_ids, logits_processor):
        # 输入 input_ids 形如 [1, L]，最后一个 token 为 initialize_tree 采样的 sample_token
        sample_token = input_ids[:, -1:].to(self.device)  # [1,1]
        context_ids = input_ids[:, 1:].to(self.device)    # 去掉首 token，与原实现一致

        scores_list = []
        parents_list = []
        ss_token = []

        len_posi = context_ids.shape[1]

        # 简易修复：不使用 past_key_values，直接以“完整前缀 + 当前分支序列”进行前向
        step_inputs = torch.cat([context_ids, sample_token], dim=1)
        out_step = self.model(input_ids=step_inputs, use_cache=False)

        # 第一层 top-k
        last_logits = out_step.logits[:, -1, :]
        last_p = torch.log_softmax(last_logits, dim=-1)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=self.device))

        ss_token.append(topk_index)
        branch_prefix = topk_index.view(-1, 1)  # [top_k,1]
        tree_mask = tree_mask_init

        for i in range(depth):
            # 与原实现保持一致的 position_ids 用于内部树掩码（小模型不使用）
            _ = len_posi + position_ids
            # 为每个分支构造批量输入：上下文 + sample_token + 分支历史
            batch_prefix = torch.cat([
                context_ids.repeat(top_k, 1),
                sample_token.repeat(top_k, 1),
                branch_prefix
            ], dim=1)
            out_br = self.model(input_ids=batch_prefix, use_cache=False)
            len_posi += 1

            last_p = torch.log_softmax(out_br.logits[:, -1, :], dim=-1)  # [top_k, vocab]
            top = torch.topk(last_p, top_k, dim=-1)
            next_indices, next_scores = top.indices, top.values

            cu_scores = next_scores + scores[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            next_tokens = next_indices.view(-1)[topk_cs_index]

            # 更新分支历史序列
            branch_prefix = torch.cat([
                branch_prefix.index_select(0, out_ids),
                next_tokens.view(-1, 1)
            ], dim=1)

            parents = (topk_cs_index + (1 + (top_k ** 2) * max(0, i - 1) + (top_k if i > 0 else 0)))
            parents_list.append(parents)
            ss_token.append(next_indices)
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], tree_mask_init), dim=3)

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token.squeeze(1), draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask_out = torch.eye(total_tokens + 1).bool()
        tree_mask_out[:, 0] = True
        for i in range(total_tokens):
            tree_mask_out[i + 1].add_(tree_mask_out[mask_index_list[i]])
        tree_position_ids = torch.sum(tree_mask_out, dim=1) - 1

        tree_mask_out = tree_mask_out.float()[None, None]
        draft_tokens = draft_tokens[None]

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()
        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth_i = position_ids_list[i]
                for j in reversed(range(depth_i + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5
            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys
            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        tree_position_ids = tree_position_ids.to(self.device)
        return draft_tokens, retrieve_indices, tree_mask_out, tree_position_ids

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig
from .adapters.deepseek_v2 import DeepseekV2HFAdapter
from .modeling_deepseek_v2_kv import DeepSeekV2KVAccessor


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        # Check if this is a DeepSeek V2 model and create adapter if needed
        self.is_deepseek_v2 = hasattr(base_model.config, 'model_type') and base_model.config.model_type == 'deepseek_v2'
        if self.is_deepseek_v2:
            self.deepseek_adapter = DeepseekV2HFAdapter(base_model)
            self.kv_accessor = DeepSeekV2KVAccessor(base_model.config, 
                                                   device=base_model.device, 
                                                   dtype=base_model.dtype)
            self.hidden_size = self.deepseek_adapter.get_hidden_size()
            self.vocab_size = self.deepseek_adapter.get_vocab_size()
        else:
            self.deepseek_adapter = None
            self.kv_accessor = None
            self.hidden_size = base_model.lm_head.weight.shape[-1]
            self.vocab_size = base_model.lm_head.weight.shape[0]
            
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral" or "DeepSeek V2"
        config = AutoConfig.from_pretrained(base_model_path)
        Type = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else None
        model_type = getattr(config, 'model_type', None)
        # NOTE 识别模型类型，但我不懂Type是什么语法
        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif model_type == 'deepseek_v2' or Type == 'DeepseekV2ForCausalLM':
            # For DeepSeek V2, use the standard HuggingFace model
            from transformers import DeepseekV2ForCausalLM
            base_model = DeepseekV2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ): # 执行base_model的一次前向传播
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
    ): # NOTE ？ 这个和 ea_generate 有啥区别啊？
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
    ): # NOTE 
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>") # 检查是否为 Llama 3 模型，并添加其专用的停止 token <|eot_id|> ？

        # 根据 temperature、top_p、top_k 是否有效构造一个 logits_processor
        # 这个对象就是把“采样/截断策略”封装成一个可对 logits 进行变换的管道
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()
        # 外部草稿适配器 KV 重置（若启用）
        if hasattr(self.ea_layer, "external_adapter") and self.ea_layer.external_adapter is not None:
            adapter = self.ea_layer.external_adapter
            if hasattr(adapter, "reset_kv"):
                adapter.reset_kv()
            else:
                adapter["stable_kv"] = None

        # NOTE  初始化 KVCache 大模型的
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values, # 每一层两个 KVCache 对象组成的列表（[K, V]），一共 num_hidden_layers 组
                past_key_values_data, # 一个“列表”，每个元素是一个五维张量，分别对应一段在同一设备上的连续层的 K/V 数据缓冲区  [2 * group_layers, batch_size(=1), num_kv_heads, max_length, head_dim
                current_length_data, # CPU 上的长整型向量，长度为 2 * num_hidden_layers（每层 K/V 各一个长度指针）  在每次接受分支后，更新为“主序列新长度”，基础模型后续前向直接基于最新长度进行增量计算
            ) = initialize_past_key_values(self.base_model,max_length=max_length) # max_length=2048
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)  
        # NOTE  1. 初始化草稿树
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        # NOTE 主生成循环
        for idx in range(max_length): 
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"): 
            # NOTE 2. tree_decoding(utils)
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states NOTE 初始化 KVCache
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    def enable_external_draft(self, draft_model_name_or_path: str, torch_dtype: torch.dtype = None):
        """
        启用外部草稿模型：加载 HF 小模型与分词器，并将适配器引用注入到 ea_layer。
        草稿模型仅用于生成树提案与稳定 KV，不依赖主模型隐藏层。
        """
        device = self.ea_layer.lm_head.weight.device
        dtype = torch_dtype if torch_dtype is not None else self.ea_layer.lm_head.weight.dtype
        small_tokenizer = _AutoTokenizerForDraft.from_pretrained(draft_model_name_or_path, use_fast=True)
        small_model = AutoModelForCausalLM.from_pretrained(draft_model_name_or_path, torch_dtype=dtype)
        adapter = ExternalDraftAdapter(small_model, small_tokenizer, device, dtype)
        self.ea_layer.external_adapter = adapter
