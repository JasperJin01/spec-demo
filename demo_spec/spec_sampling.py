import math
from typing import List, Tuple, Optional, Dict, Callable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TemperatureLogitsWarper, TopPLogitsWarper, LogitsProcessorList

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


def propose_tree_kv(
    draft_model: AutoModelForCausalLM,
    draft_past: Tuple,
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
    stream_handler: Optional[Callable[[str], None]] = None,
    is_llama3: bool = False,
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

    # 停止 token 集合：EOS + （若是 Llama3）EOT
    stop_token_ids = [eos_token_id]
    if is_llama3:
        try:
            eot_id = base_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != -1:
                stop_token_ids.append(eot_id)
        except Exception:
            pass

    generated_text: List[str] = []

    # 统一的流式输出接口
    def emit(text_chunk: str):
        if not text_chunk:
            return
        generated_text.append(text_chunk)
        if stream_handler is not None:
            try:
                stream_handler(text_chunk)
            except Exception:
                pass
        else:
            # 默认直接输出到 stdout
            import sys
            sys.stdout.write(text_chunk)
            sys.stdout.flush()

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
            accepted_ids = best[:accept_len].tolist()
            # 提前检测 stop token 的位置（优先截断）
            stop_pos = None
            for i, tid in enumerate(accepted_ids):
                if tid in stop_token_ids:
                    stop_pos = i
                    break
            if stop_pos is not None:
                accepted_ids = accepted_ids[:stop_pos + 1]
            decoded = base_tokenizer.decode(accepted_ids, skip_special_tokens=True)
            emit(decoded)

            # 批量推进 draft 的缓存到接受后的状态
            accepted_tensor = best[:accept_len]
            draft_logits_chunk, draft_past = _forward_chunk_with_cache(draft_model, accepted_tensor, draft_past, device)
            draft_last_logits = draft_logits_chunk[:, -1, :]
            # 更新 base 的缓存到新状态（由校验函数已计算）
            base_past, base_last_logits = new_base_past, new_base_logits

            # 更新 warper 的 current_input_ids 视图
            base_curr_ids = torch.cat([base_curr_ids, accepted_tensor.unsqueeze(0)], dim=1)
            draft_curr_ids = torch.cat([draft_curr_ids, accepted_tensor.unsqueeze(0)], dim=1)

            # 若最后一个为 stop token，则结束
            if accepted_ids and accepted_ids[-1] in stop_token_ids:
                return "".join(generated_text)

        # 处理不匹配：插入 base 的预测 token（按分布随机采样）
        if mismatch_token is not None:
            decoded_mis = base_tokenizer.decode([mismatch_token], skip_special_tokens=True)
            emit(decoded_mis)

            # 推进两个模型的 KVCache
            draft_last_logits, draft_past = _forward_next_with_cache(draft_model, mismatch_token, draft_past, device)
            base_last_logits, base_past = _forward_next_with_cache(base_model, mismatch_token, base_past, device)

            # 更新 warper 的 current_input_ids 视图
            token_tensor = torch.tensor([[mismatch_token]], device=device)
            base_curr_ids = torch.cat([base_curr_ids, token_tensor], dim=1)
            draft_curr_ids = torch.cat([draft_curr_ids, token_tensor], dim=1)

            # stop 则结束
            if mismatch_token in stop_token_ids:
                return "".join(generated_text)

        # 若既没有接受也没有不匹配（极端情况），则从 base 分布采 1 个 token 以推进
        if accept_len == 0 and mismatch_token is None:
            warped = base_warper(base_curr_ids, base_last_logits) if base_warper is not None else base_last_logits
            probs = torch.softmax(warped, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1, replacement=True)
            next_token = int(sampled.item())
            decoded_token = base_tokenizer.decode([next_token], skip_special_tokens=True)
            emit(decoded_token)

            draft_last_logits, draft_past = _forward_next_with_cache(draft_model, next_token, draft_past, device)
            base_last_logits, base_past = _forward_next_with_cache(base_model, next_token, base_past, device)

            tok_tensor = torch.tensor([[next_token]], device=device)
            base_curr_ids = torch.cat([base_curr_ids, tok_tensor], dim=1)
            draft_curr_ids = torch.cat([draft_curr_ids, tok_tensor], dim=1)

            if next_token in stop_token_ids:
                return "".join(generated_text)

    # 循环结束，返回累计文本
    return "".join(generated_text)