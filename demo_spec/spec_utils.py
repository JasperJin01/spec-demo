import os
import math
from typing import List, Tuple, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
)


def _maybe_set_llama3_chat_template(tokenizer: AutoTokenizer) -> None:
    try:
        tpl = getattr(tokenizer, "chat_template", None)
        name = getattr(tokenizer, "name_or_path", "") or ""
        if not tpl:
            lower = name.lower()
            if ("llama-3" in lower) or ("meta-llama-3" in lower) or ("llama3" in lower):
                tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    except Exception:
        pass


def setup_device_and_dtype(prefer_fp16: bool = True) -> Tuple[torch.device, torch.dtype]:
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
            import os
            has_tokenizer_json = os.path.isfile(os.path.join(source, "tokenizer.json")) if os.path.isdir(source) else False
            has_tokenizer_model = os.path.isfile(os.path.join(source, "tokenizer.model")) if os.path.isdir(source) else False
            raise RuntimeError(
                f"Failed to load tokenizer from '{source}'. "
                f"Ensure the directory contains tokenizer.json (Fast) or tokenizer.model (SentencePiece), or pass a valid HF model id. "
                f"Present files: tokenizer.json={has_tokenizer_json}, tokenizer.model={has_tokenizer_model}. Details: {type(e2).__name__}: {e2}"
            )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _maybe_set_llama3_chat_template(tokenizer)
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
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
    return logits


def topk_tokens(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=k, dim=-1)
    topk_ids = topk.indices.squeeze(0)
    topk_probs = topk.values.squeeze(0)
    topk_logps = torch.log(topk_probs + 1e-12)
    return topk_ids, topk_logps


def propose_tree(
    draft_model: AutoModelForCausalLM,
    draft_tokenizer: AutoTokenizer,
    cur_ids: torch.Tensor,
    top_k: int,
    depth: int,
) -> List[Dict[str, torch.Tensor]]:
    device = cur_ids.device
    candidates: List[Dict[str, torch.Tensor]] = []
    logits_0 = last_logits(draft_model, cur_ids)
    first_ids, first_logps = topk_tokens(logits_0, k=top_k)
    for i in range(first_ids.shape[0]):
        path_tokens = []
        path_score = 0.0
        t1 = first_ids[i].item()
        path_tokens.append(t1)
        path_score += first_logps[i].item()
        branch_ids = torch.cat([cur_ids, torch.tensor([[t1]], device=device)], dim=1)
        for d in range(1, depth):
            logits_d = last_logits(draft_model, branch_ids)
            probs_d = torch.softmax(logits_d, dim=-1)
            top1_prob, top1_id = torch.max(probs_d, dim=-1)
            t_next = top1_id.item()
            path_tokens.append(t_next)
            path_score += math.log(top1_prob.item() + 1e-12)
            branch_ids = torch.cat([branch_ids, torch.tensor([[t_next]], device=device)], dim=1)
        candidates.append({
            "tokens": torch.tensor(path_tokens, dtype=torch.long, device=device),
            "score": torch.tensor(path_score, dtype=torch.float32, device=device),
        })
    candidates.sort(key=lambda c: c["score"].item(), reverse=True)
    return candidates


def _render_messages_with_template(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        chat_tpl = getattr(tokenizer, "chat_template", None)
        if chat_tpl:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        pass
    lines = []
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