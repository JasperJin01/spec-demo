import argparse
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=None,  # 禁用 accelerate 自动分片
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    # 支持流式输出，便于观察模型是否成功跑起来
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    # 将输入张量移动到模型第一个设备
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
            use_cache=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Run DeepSeek-V2-Lite locally with transformers")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data1/jinjm_data/ktransformers/DeepSeek-V2-Lite",
        help="Local path to DeepSeek-V2-Lite model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好！请用简洁中文自我介绍。",
        help="Prompt text for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )

    args = parser.parse_args()

    try:
        tokenizer, model = load_model(args.model_path)
    except Exception as e:
        print(f"[Error] 模型加载失败: {e}", file=sys.stderr)
        print("请确认 transformers、torch 已安装，且模型路径正确。", file=sys.stderr)
        sys.exit(1)

    print("模型加载完成，开始生成...\n")
    generate(model, tokenizer, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()