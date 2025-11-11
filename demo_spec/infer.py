#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的多轮对话脚本：
- 使用 transformers 自动加载模型和分词器
- 支持多轮对话历史，采用 chat template（若模型提供）
- 运行：python demo_spec/infer.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  退出：输入 :q 或 Ctrl+C
"""

import argparse
import os
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def build_plain_prompt(messages):
    """当模型不提供 chat_template 时的降级提示拼接。"""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
    parts.append("Assistant:")
    return "".join(parts)


def chat_loop(model_id: str, system_prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 尝试使用更快的 dtype（GPU 上 bfloat16 / float16），CPU 上使用 float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    print(f"[INFO] 加载模型: {model_id} (device={device}, dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # 生成参数
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print("\n开始对话，输入 :q 退出。\n")
    try:
        while True:
            user_text = input("用户: ").strip()
            if user_text == "" or user_text == ":q":
                print("[INFO] 对话结束。")
                break
            messages.append({"role": "user", "content": user_text})

            # 使用 chat template（如果可用）
            if getattr(tokenizer, "chat_template", None):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = build_plain_prompt(messages)

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            # 仅解码新生成的 tokens
            new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
            reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # 有些模型会生成多余的前后缀，简单清理
            reply = reply.replace("Assistant:", "").strip()

            print(f"助手: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\n[INFO] 已中断，对话结束。")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Transformers 多轮对话")
    parser.add_argument("--model", type=str, default="/home/jinjm/dev/models/llama31-8b-chat/LLM-Research/Meta-Llama-3___1-8B-Instruct", help="HuggingFace 模型名称或本地路径")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="系统提示词")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="单次回复的最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="核采样概率阈值")
    args = parser.parse_args()

    chat_loop(
        model_id=args.model,
        system_prompt=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )