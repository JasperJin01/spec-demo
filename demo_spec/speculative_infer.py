#!/usr/bin/env python3
"""
Speculative inference demo using Transformers with a draft tree approach.
This file now serves as the CLI entrypoint and orchestrates calls into split modules.
"""

import argparse

from spec_core import setup_device_and_dtype, load_model_and_tokenizer
from spec_sampling import generate_speculative_stream


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative decoding demo (tree-based) with optional random sampling.")
    # 不要改我的参数默认值！
    parser.add_argument("--base-model", help="HF model id or local path for the base (target) model",
                        default="/home/jinjm/dev/models/llama31-8b-chat/LLM-Research/Meta-Llama-3___1-8B-Instruct")
    parser.add_argument("--draft-model", help="HF model id or local path for the draft (proposal) model",
                        default="/home/jinjm/dev/models/Llama-3.2-1B")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--top-k", type=int, default=4, help="Tree width for draft proposals")
    parser.add_argument("--depth", type=int, default=4, help="Tree depth for draft proposals")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature (None or 1.0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (0<p<1 to enable)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow loading models with custom code")
    parser.add_argument("--no-fp16", action="store_true", help="Force float32 (disable fp16)")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="系统提示词")

    args = parser.parse_args()

    device, dtype = setup_device_and_dtype(prefer_fp16=not args.no_fp16)

    base_model, base_tokenizer = load_model_and_tokenizer(
        args.base_model, device, dtype, trust_remote_code=args.trust_remote_code
    )
    draft_model, draft_tokenizer = load_model_and_tokenizer(
        args.draft_model, device, dtype, trust_remote_code=args.trust_remote_code
    )

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    print("\n开始对话，输入 :q 退出。\n")
    try:
        while True:
            user_text = input("用户: ").strip()
            if user_text == "" or user_text == ":q":
                print("[INFO] 对话结束。")
                break
            messages.append({"role": "user", "content": user_text})

            reply = generate_speculative_stream(
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                draft_model=draft_model,
                draft_tokenizer=draft_tokenizer,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                depth=args.depth,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            # 流式输出已在函数内部完成，这里补一个换行并记录到历史
            print()
            messages.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\n[INFO] 已中断，对话结束。")
  