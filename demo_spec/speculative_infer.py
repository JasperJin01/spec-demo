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
from transformers import AutoTokenizer, AutoModelForCausalLM

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
) -> List[Dict[str, torch.Tensor]]:
    # 首步 top‑k + 后续贪心
    probs0 = torch.softmax(draft_last_logits, dim=-1)
    topk = torch.topk(probs0, k=top_k, dim=-1)
    first_ids = topk.indices.squeeze(0)
    first_probs = topk.values.squeeze(0)

    candidates: List[Dict[str, torch.Tensor]] = []
    for i in range(first_ids.shape[0]):
        path_tokens: List[int] = []
        path_score = float(torch.log(first_probs[i] + 1e-12).item())
        t1 = int(first_ids[i].item())
        path_tokens.append(t1)
        # 复制当前 past 的引用，逐步扩展
        tmp_past = draft_past
        tmp_logits = draft_last_logits
        # 将第一步 token 前向以更新 past
        tmp_logits, tmp_past = _forward_next_with_cache(draft_model, t1, tmp_past, device)
        # 继续贪心扩展至 depth
        for d in range(1, depth):
            probs_d = torch.softmax(tmp_logits, dim=-1)
            top1_prob, top1_id = torch.max(probs_d, dim=-1)  # [1]
            t_next = int(top1_id.item())
            path_tokens.append(t_next)
            path_score += math.log(top1_prob.item() + 1e-12)
            tmp_logits, tmp_past = _forward_next_with_cache(draft_model, t_next, tmp_past, device)

        candidates.append({
            "tokens": torch.tensor(path_tokens, dtype=torch.long, device=device),
            "score": torch.tensor(path_score, dtype=torch.float32, device=device),
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
) -> Tuple[int, Optional[int], Tuple, torch.Tensor]:
    # 并行验证：一次性前向获取整段候选的预测，向量化比较得到接受长度
    accept_len = 0
    mismatch_token: Optional[int] = None

    if candidate_tokens.numel() == 0:
        return 0, None, base_past, base_last_logits

    # 第一个 token 的校验使用已有的 base_last_logits（无需前向）
    first_pred = torch.argmax(base_last_logits, dim=-1).item()
    c1 = candidate_tokens[0].item()
    if first_pred != c1:
        mismatch_token = first_pred
        return 0, mismatch_token, base_past, base_last_logits

    # 接受首 token
    accept_len = 1

    L = candidate_tokens.shape[0]
    if L > 1:
        # 一次性前向整段候选，取每个位置的预测用于比较 c2..cL
        chunk_logits, _ = _forward_chunk_with_cache(base_model, candidate_tokens, base_past, device)
        # 位置 i-1 的 logits 是在消费了 c1..c(i-1) 之后对下一 token 的预测
        for i in range(2, L + 1):
            pred_i = torch.argmax(chunk_logits[:, i - 2, :], dim=-1).item()
            ci = candidate_tokens[i - 1].item()
            if pred_i == ci:
                accept_len += 1
            else:
                mismatch_token = pred_i
                break

    # 统一用一次前向更新到“已接受长度”的 KVCache 与最后一步 logits
    accepted_seq = candidate_tokens[:accept_len]
    update_logits, update_past = _forward_chunk_with_cache(base_model, accepted_seq, base_past, device)
    new_last_logits = update_logits[:, -1, :]

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

    base_last_logits, base_past = _init_with_cache(base_model, base_ids)
    draft_last_logits, draft_past = _init_with_cache(draft_model, draft_ids)

    if eos_token_id is None:
        eos_token_id = base_tokenizer.eos_token_id

    generated_text = []

    for step in range(max_new_tokens):
        # 草稿模型提出树
        candidates = propose_tree_kv(draft_model, draft_past, draft_last_logits, top_k=top_k, depth=depth, device=device)
        best = candidates[0]["tokens"]

        # 用 base 并行校验并返回“已接受长度”的新 KV 与 logits
        accept_len, mismatch_token, new_base_past, new_base_logits = verify_with_base_kv(
            base_model, base_past, base_last_logits, best, device
        )

        # 接受的 token：流式输出；KVCache 批量一次性推进
        if accept_len > 0:
            accepted = best[:accept_len].tolist()
            # 批量 decode 后一次性打印，避免子词级重复的视觉噪声
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

            # 若最后一个为 EOS，则结束
            if accepted[-1] == eos_token_id:
                return "".join(generated_text)

        # 处理不匹配：插入 base 的预测 token（长度 1 的批量推进）
        if mismatch_token is not None:
            decoded_mis = base_tokenizer.decode([mismatch_token], skip_special_tokens=True)
            sys.stdout.write(decoded_mis)
            sys.stdout.flush()
            generated_text.append(decoded_mis)

            single = torch.tensor([mismatch_token], dtype=torch.long, device=device)
            # 批量推进 draft 与 base 的缓存（单 token）
            draft_logits_chunk, draft_past = _forward_chunk_with_cache(draft_model, single, draft_past, device)
            draft_last_logits = draft_logits_chunk[:, -1, :]
            base_logits_chunk, base_past = _forward_chunk_with_cache(base_model, single, base_past, device)
            base_last_logits = base_logits_chunk[:, -1, :]

            if mismatch_token == eos_token_id:
                return "".join(generated_text)

    return "".join(generated_text)


def main():
    parser = argparse.ArgumentParser(description="Speculative inference demo with a draft tree (Transformers)")
    parser.add_argument("--base-model", type=str, required=False, 
                        default="/home/jinjm/dev/models/llama31-8b-chat/LLM-Research/Meta-Llama-3___1-8B-Instruct", help="HF model id/path for the big (target) model")
    parser.add_argument("--draft-model", type=str, required=False, 
                        default="/home/jinjm/dev/models/Llama-3.2-1B", help="HF model id/path for the draft (assistant) model")
    # parser.add_argument("--base-model", type=str, required=False, 
    #                     default="/home/jinjm/dev/models/Llama-3.1-8B-unsloth-bnb-4bit", help="HF model id/path for the big (target) model")
    # parser.add_argument("--draft-model", type=str, required=False, 
    #                     default="/data1/jinjm_data/dev/models/Llama-3.2-1B-bnb-4bit", help="HF model id/path for the draft (assistant) model")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=4, help="Tree width for draft proposals")
    parser.add_argument("--depth", type=int, default=4, help="Tree depth (length of drafted path)")
    parser.add_argument("--fp16", action="store_true", help="Prefer fp16 on CUDA")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code when needed (HF)")
    parser.add_argument("--base-tokenizer", type=str, default=None, help="Optional tokenizer source for base model (path or HF id)")
    parser.add_argument("--draft-tokenizer", type=str, default=None, help="Optional tokenizer source for draft model (path or HF id)")
    args = parser.parse_args()

    device, dtype = setup_device_and_dtype(prefer_fp16=args.fp16)

    # 加载模型和分词器
    print(f"Loading base model: {args.base_model}")
    base_model, base_tokenizer = load_model_and_tokenizer(model_id=args.base_model, device=device, dtype=dtype, trust_remote_code=args.trust_remote_code, tokenizer_source=args.base_tokenizer)
    print(f"Loading draft model: {args.draft_model}")
    draft_tok_src = args.draft_tokenizer or args.base_tokenizer
    draft_model, draft_tokenizer = load_model_and_tokenizer(model_id=args.draft_model, device=device, dtype=dtype, trust_remote_code=args.trust_remote_code, tokenizer_source=draft_tok_src)

    # 进入多轮对话：从标准输入读取，每轮生成时进行流式输出
    print("进入多轮对话模式，输入 'exit' 或 'quit' 结束。")
    history: List[Tuple[str, str]] = []
    while True:
        try:
            user_msg = input("用户> ").strip()
        except EOFError:
            break
        if user_msg.lower() in ("exit", "quit"):
            break
        if not user_msg:
            continue
        history.append(("user", user_msg))

        # 使用官方聊天模板：由 tokenizer.chat_template 决定
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for role, content in history:
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
        # 提示词将由 generate_speculative_stream 内部根据各自 tokenizer 渲染

        print("助手> ", end="", flush=True)
        gen_text = generate_speculative_stream(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            depth=args.depth,
        )
        print()  # 换行
        history.append(("assistant", gen_text))

    # 结束后不再打印未定义变量
    # print("\n=== Output ===")
    # print(text)


if __name__ == "__main__":
    main()