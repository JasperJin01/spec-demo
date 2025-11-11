#!/usr/bin/env python3
"""
Speculative inference demo using Transformers with a draft tree approach.
- Loads two models: a big (target) model and a smaller draft model
- Uses the draft model to propose a top-k branching tree with configurable depth
- Verifies and accepts drafted tokens using the big model following a speculative decoding rule

This demo intentionally avoids dependencies on the eagle folder, but takes inspiration
from its tree-based drafting idea (top_k width, greedy depth expansion).

Note: This is a demonstration focusing on clarity over raw performance. For simplicity,
we rebuild model states from the full sequence per step, which is slower than using
past_key_values efficiently. It should work with common HF causal LMs (e.g., gpt2).
"""

import argparse
import math
import sys
import os
from typing import List, Tuple, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TemperatureLogitsWarper, TopPLogitsWarper, LogitsProcessorList

# 注入官方 Llama 3/3.1/3.2 Instruct 的 chat_template（Jinja），当 tokenizer.chat_template 缺失时使用
LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
)


def _maybe_set_llama3_chat_template(tokenizer: AutoTokenizer) -> None:
    try:
        tpl = getattr(tokenizer, "chat_template", None)
        name = getattr(tokenizer, "name_or_path", "") or ""
        if not tpl:
            # 仅当模型名称包含 Llama-3 / Meta-Llama-3 等标识时设置
            lower = name.lower()
            if ("llama-3" in lower) or ("meta-llama-3" in lower) or ("llama3" in lower):
                tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    except Exception:
        # 静默失败，保持兼容
        pass


def setup_device_and_dtype(prefer_fp16: bool = True) -> Tuple[torch.device, torch.dtype]:
    """Pick device and dtype for inference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16 if prefer_fp16 else torch.float32
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype


def load_model_and_tokenizer(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
    tokenizer_source: Optional[str] = None,
):
    source = tokenizer_source or model_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, legacy=False, trust_remote_code=trust_remote_code)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, trust_remote_code=trust_remote_code)
        except Exception as e2:
            has_tokenizer_json = os.path.isfile(os.path.join(source, "tokenizer.json")) if os.path.isdir(source) else False
            has_tokenizer_model = os.path.isfile(os.path.join(source, "tokenizer.model")) if os.path.isdir(source) else False
            raise RuntimeError(
                f"Failed to load tokenizer from '{source}'. "
                f"Ensure the directory contains tokenizer.json (Fast) or tokenizer.model (SentencePiece), or pass a valid HF model id. "
                f"Present files: tokenizer.json={has_tokenizer_json}, tokenizer.model={has_tokenizer_model}. Details: {type(e2).__name__}: {e2}"
            )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 当 chat_template 缺失且模型为 Llama3 系列时，设置官方模板
    _maybe_set_llama3_chat_template(tokenizer)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model, tokenizer


def last_logits(model: AutoModelForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Run the model on the full sequence and return logits for the last position."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]  # [batch=1, vocab]
    return logits


def topk_tokens(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-k token ids and log-probabilities from logits."""
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=k, dim=-1)
    topk_ids = topk.indices.squeeze(0)
    topk_probs = topk.values.squeeze(0)
    # Convert to log-probs for additive scoring across steps
    topk_logps = torch.log(topk_probs + 1e-12)
    return topk_ids, topk_logps


def propose_tree(
    draft_model: AutoModelForCausalLM,
    draft_tokenizer: AutoTokenizer,
    cur_ids: torch.Tensor,
    top_k: int,
    depth: int,
) -> List[Dict[str, torch.Tensor]]:
    """Propose a draft tree of width top_k and greedy depth expansion.
    Returns a list of candidate dicts: {"tokens": Tensor([t1, t2, ...]), "score": float_logp}
    
    Strategy:
    - From current sequence, get next-token top-k using draft logits
    - For each top-1 token at the first step, greedily expand depth-1 using top-1 each step
      to form a candidate path of length=depth
    - Score is sum of log-probs along the path
    """
    device = cur_ids.device
    candidates: List[Dict[str, torch.Tensor]] = []

    # First step: top-k branching
    logits_0 = last_logits(draft_model, cur_ids)
    first_ids, first_logps = topk_tokens(logits_0, k=top_k)

    for i in range(first_ids.shape[0]):
        path_tokens = []
        path_score = 0.0
        # Step 1 token
        t1 = first_ids[i].item()
        path_tokens.append(t1)
        path_score += first_logps[i].item()
        # Greedy depth expansion for subsequent steps
        branch_ids = torch.cat([cur_ids, torch.tensor([[t1]], device=device)], dim=1)
        for d in range(1, depth):
            logits_d = last_logits(draft_model, branch_ids)
            probs_d = torch.softmax(logits_d, dim=-1)
            top1_prob, top1_id = torch.max(probs_d, dim=-1)  # [1]
            t_next = top1_id.item()
            path_tokens.append(t_next)
            path_score += math.log(top1_prob.item() + 1e-12)
            branch_ids = torch.cat([branch_ids, torch.tensor([[t_next]], device=device)], dim=1)
        candidates.append({
            "tokens": torch.tensor(path_tokens, dtype=torch.long, device=device),
            "score": torch.tensor(path_score, dtype=torch.float32, device=device),
        })

    # Sort candidates by score descending (best first)
    candidates.sort(key=lambda c: c["score"].item(), reverse=True)
    return candidates


def verify_with_base(
    base_model: AutoModelForCausalLM,
    base_tokenizer: AutoTokenizer,
    cur_ids: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> Tuple[int, Optional[int]]:
    """Verify candidate tokens with base model following speculative decoding rules.
    Returns (accept_len, mismatch_token) where mismatch_token is None if fully accepted.

    Implementation detail: to keep it simple, we recompute logits from the full sequence
    after each acceptance, which is slower but correct for demonstration.
    """
    device = cur_ids.device
    accept_len = 0
    mismatch_token: Optional[int] = None

    temp_ids = cur_ids.clone()
    for j in range(candidate_tokens.shape[0]):
        # Compute next-token prediction by base model
        logits = last_logits(base_model, temp_ids)
        base_next = torch.argmax(logits, dim=-1).item()
        drafted = candidate_tokens[j].item()
        if base_next == drafted:
            # Accept this drafted token and move forward
            accept_len += 1
            temp_ids = torch.cat([temp_ids, torch.tensor([[drafted]], device=device)], dim=1)
        else:
            # Mismatch: return the base token to insert instead
            mismatch_token = base_next
            break

    return accept_len, mismatch_token


def propose_tree_kv(
    draft_model: AutoModelForCausalLM,
    draft_past: Tuple, # KVCache
    draft_last_logits: torch.Tensor,
    top_k: int,
    depth: int,
    device: torch.device,
    logits_warper: Optional[LogitsProcessorList] = None,
    current_input_ids: Optional[torch.Tensor] = None,
) -> List[Dict[str, torch.Tensor]]:
    # 引入随机采样：首步按 warper(temperature/top_p) 后的分布做无放回采样 K 个；后续每步按同分布采 1 个
    if logits_warper is not None:
        warped0 = logits_warper(current_input_ids if current_input_ids is not None else torch.zeros((1,1), dtype=torch.long, device=device), draft_last_logits)
    else:
        warped0 = draft_last_logits
    probs0 = torch.softmax(warped0, dim=-1)
    # 若 top_p 造成有效概率项不足，则回退为有放回采样或取最大值
    support = (probs0 > 0).sum(dim=-1).item()
    k_eff = max(1, min(top_k, int(support)))
    if k_eff >= top_k and support >= top_k:
        first_ids = torch.multinomial(probs0, num_samples=top_k, replacement=False)
    elif support >= 1:
        first_ids = torch.multinomial(probs0, num_samples=k_eff, replacement=True)
    else:
        # 极端情况下（全部 0），回退到 argmax
        first_ids = torch.argmax(probs0, dim=-1, keepdim=True)
    first_probs = torch.gather(probs0, -1, first_ids)

    candidates: List[Dict[str, torch.Tensor]] = []
    for i in range(first_ids.shape[-1]):
        path_tokens: List[int] = []
        proposal_probs: List[float] = []
        path_score = float(torch.log(first_probs[:, i] + 1e-12).item())
        t1 = int(first_ids[:, i].item())
        path_tokens.append(t1)
        proposal_probs.append(float(first_probs[:, i].item()))
        # 复制当前 past 的引用，逐步扩展
        tmp_past = draft_past
        tmp_logits = draft_last_logits
        # 将第一步 token 前向以更新 past
        tmp_logits, tmp_past = _forward_next_with_cache(draft_model, t1, tmp_past, device)
        # 继续随机采样扩展至 depth
        for d in range(1, depth):
            if logits_warper is not None:
                warped_d = logits_warper(current_input_ids if current_input_ids is not None else torch.zeros((1,1), dtype=torch.long, device=device), tmp_logits)
            else:
                warped_d = tmp_logits
            probs_d = torch.softmax(warped_d, dim=-1)
            sampled_id = torch.multinomial(probs_d, num_samples=1, replacement=True)
            sampled_prob = torch.gather(probs_d, -1, sampled_id)
            t_next = int(sampled_id.item())
            path_tokens.append(t_next)
            proposal_probs.append(float(sampled_prob.item()))
            path_score += math.log(sampled_prob.item() + 1e-12)
            tmp_logits, tmp_past = _forward_next_with_cache(draft_model, t_next, tmp_past, device)

        candidates.append({
            "tokens": torch.tensor(path_tokens, dtype=torch.long, device=device),
            "score": torch.tensor(path_score, dtype=torch.float32, device=device),
            "proposal_probs": torch.tensor(proposal_probs, dtype=torch.float32, device=device),
        })

    candidates.sort(key=lambda c: c["score"].item(), reverse=True)
    return candidates

# 将提示词 tokenize，做第一次前向，拿到“最后一步 logits”和“KVCache”
def _init_with_cache(model: AutoModelForCausalLM, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def _forward_next_with_cache(model: AutoModelForCausalLM, token_id: int, past_key_values: Tuple, device: torch.device) -> Tuple[torch.Tensor, Tuple]:
    next_ids = torch.tensor([[token_id]], device=device)
    with torch.no_grad():
        out = model(input_ids=next_ids, use_cache=True, past_key_values=past_key_values)
    return out.logits[:, -1, :], out.past_key_values


def _forward_chunk_with_cache(
    model: AutoModelForCausalLM,
    token_ids_seq: torch.Tensor,
    past_key_values: Tuple,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple]:
    if token_ids_seq.ndim == 1:
        token_ids_seq = token_ids_seq.unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=token_ids_seq.to(device), use_cache=True, past_key_values=past_key_values)
    return out.logits, out.past_key_values


# 使用 KVCache 的校验过程：依据 base_last_logits 的下一个 token 进行对齐

def verify_with_base_kv(
    base_model: AutoModelForCausalLM,
    base_past: Tuple,
    base_last_logits: torch.Tensor,
    candidate_tokens: torch.Tensor,
    device: torch.device,
    logits_warper: Optional[LogitsProcessorList] = None,
    candidate_probs: Optional[torch.Tensor] = None,
) -> Tuple[int, Optional[int], Tuple, torch.Tensor]:
    # EAGLE3 风格的随机接受：按 px/qx 的比值做随机接受测试；拒绝时从 base 分布采样 mismatch_token
    accept_len = 0
    mismatch_token: Optional[int] = None

    if candidate_tokens.numel() == 0:
        return 0, None, base_past, base_last_logits

    L = candidate_tokens.shape[0]

    # 先一次性前向整段候选，取每个位置的 base logits
    chunk_logits, _ = _forward_chunk_with_cache(base_model, candidate_tokens, base_past, device)

    # 遍历每个位置 i=1..L，计算 base 概率 px 与 draft 提案概率 qx，进行随机接受
    for i in range(1, L + 1):
        if i == 1:
            cur_logits = base_last_logits
        else:
            cur_logits = chunk_logits[:, i - 2, :]
        if logits_warper is not None:
            warped = logits_warper(torch.zeros((1,1), dtype=torch.long, device=device), cur_logits)
        else:
            warped = cur_logits
        probs = torch.softmax(warped, dim=-1)
        ci = candidate_tokens[i - 1].item()
        px = float(probs[0, ci].item())
        qx = float(candidate_probs[i - 1].item()) if (candidate_probs is not None) else 1e-12
        u = float(torch.rand(1).item())
        thresh = min(1.0, px / (qx + 1e-12))
        if u <= thresh:
            accept_len += 1
            continue
        else:
            # 随机从 base 的分布采样一个 mismatch token
            sampled = torch.multinomial(probs, num_samples=1, replacement=True)
            mismatch_token = int(sampled.item())
            break

    # 统一用一次前向更新到“已接受长度”的 KVCache 与最后一步 logits
    if accept_len > 0:
        accepted_seq = candidate_tokens[:accept_len]
        update_logits, update_past = _forward_chunk_with_cache(base_model, accepted_seq, base_past, device)
        new_last_logits = update_logits[:, -1, :]
    else:
        update_past = base_past
        new_last_logits = base_last_logits

    return accept_len, mismatch_token, update_past, new_last_logits


# 流式生成（KVCache），每接受或插入一个 token 立即输出

def _render_messages_with_template(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    # 优先使用 tokenizer.chat_template； 若未设置或报错，则回退到简易格式
    try:
        chat_tpl = getattr(tokenizer, "chat_template", None)
        if chat_tpl:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        pass
    # Fallback: 简易的人类可读模板
    lines = []
    # 取第一个 system
    for m in messages:
        if m.get("role") == "system":
            lines.append(f"System: {m.get('content','')}\n")
            break
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            lines.append(f"User: {content}\n")
        elif role == "assistant":
            lines.append(f"Assistant: {content}\n")
    lines.append("Assistant: ")
    return "".join(lines)


def generate_speculative_stream(
    base_model: AutoModelForCausalLM,
    base_tokenizer: AutoTokenizer,
    draft_model: AutoModelForCausalLM,
    draft_tokenizer: AutoTokenizer,
    messages: Optional[List[Dict[str, str]]] = None,
    prompt: Optional[str] = None,
    max_new_tokens: int = 128,
    top_k: int = 4,
    depth: int = 4,
    eos_token_id: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    device = next(base_model.parameters()).device
    # 根据提供的 messages 使用各自 tokenizer 的聊天模板渲染；否则使用原始 prompt
    if messages is not None:
        base_prompt = _render_messages_with_template(base_tokenizer, messages)
        draft_prompt = _render_messages_with_template(draft_tokenizer, messages)
    else:
        base_prompt = prompt or ""
        draft_prompt = prompt or ""

    # 初始化 past 与最后一步 logits
    base_ids = base_tokenizer(base_prompt, return_tensors="pt").input_ids.to(device)
    draft_ids = draft_tokenizer(draft_prompt, return_tensors="pt").input_ids.to(device)

    # 构造 warper（仅在提供了 temperature/top_p 时启用）
    base_warper = None
    draft_warper = None
    if (temperature is not None and temperature > 0 and temperature != 1.0) or (top_p is not None and 0 < top_p < 1.0):
        base_warper = LogitsProcessorList([])
        draft_warper = LogitsProcessorList([])
        if temperature is not None and temperature > 0 and temperature != 1.0:
            base_warper.append(TemperatureLogitsWarper(temperature))
            draft_warper.append(TemperatureLogitsWarper(temperature))
        if top_p is not None and 0 < top_p < 1.0:
            base_warper.append(TopPLogitsWarper(top_p))
            draft_warper.append(TopPLogitsWarper(top_p))

    # 追踪当前的 input_ids（仅用于 warper 的接口），随着已接受/插入推进
    base_curr_ids = base_ids.clone()
    draft_curr_ids = draft_ids.clone()

    # 初始化 past 与最后一步 logits（在构造 warper 之后进行）
    base_last_logits, base_past = _init_with_cache(base_model, base_curr_ids)
    draft_last_logits, draft_past = _init_with_cache(draft_model, draft_curr_ids)

    if eos_token_id is None:
        eos_token_id = base_tokenizer.eos_token_id

    generated_text = []

    for step in range(max_new_tokens):
        # 草稿模型提出树（随机采样）
        candidates = propose_tree_kv(
            draft_model, draft_past, draft_last_logits, top_k=top_k, depth=depth, device=device,
            logits_warper=draft_warper, current_input_ids=draft_curr_ids
        )
        best = candidates[0]["tokens"]
        best_q = candidates[0]["proposal_probs"]

        # 用 base 并行校验（随机接受），并返回“已接受长度”的新 KV 与 logits
        accept_len, mismatch_token, new_base_past, new_base_logits = verify_with_base_kv(
            base_model, base_past, base_last_logits, best, device,
            logits_warper=base_warper, candidate_probs=best_q
        )

        # 接受的 token：流式输出；KVCache 批量一次性推进
        if accept_len > 0:
            accepted = best[:accept_len].tolist()
            decoded = base_tokenizer.decode(accepted, skip_special_tokens=True)
            sys.stdout.write(decoded)
            sys.stdout.flush()
            generated_text.append(decoded)

            # 批量推进 draft 的缓存到接受后的状态
            accepted_tensor = best[:accept_len]
            draft_logits_chunk, draft_past = _forward_chunk_with_cache(draft_model, accepted_tensor, draft_past, device)
            draft_last_logits = draft_logits_chunk[:, -1, :]
            # 更新 base 的缓存到新状态（由校验函数已计算）
            base_past, base_last_logits = new_base_past, new_base_logits

            # 更新 warper 的 current_input_ids 视图
            base_curr_ids = torch.cat([base_curr_ids, accepted_tensor.unsqueeze(0)], dim=1)
            draft_curr_ids = torch.cat([draft_curr_ids, accepted_tensor.unsqueeze(0)], dim=1)

            # 若最后一个为 EOS，则结束
  