import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import argparse
import sys
import time

import torch

try:
    from ..model.ea_model import EaModel
except Exception:
    from eagle.model.ea_model import EaModel

try:
    from fastchat.model import get_conversation_template
except Exception:
    get_conversation_template = None


def warmup(model, args):
    """Warmup the model with a simple generation to initialize buffers."""
    # Skip warmup for LLaMA3 models to avoid vocabulary mismatch issues
    if args.model_type == "llama-3-instruct":
        print("跳过LLaMA3模型的warmup以避免词汇表不匹配问题")
        return
        
    # Exactly like webui.py warmup function
    conv = get_conversation_template(args.model_type)

    if args.model_type == "llama-2-chat":
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
    elif args.model_type == "mixtral":
        conv = get_conversation_template("llama-2-chat")
        conv.system_message = ''
        conv.sep2 = "</s>"
    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if args.model_type == "llama-2-chat":
        prompt += " "
    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    try:
        for output_ids in model.ea_generate(input_ids):
            ol = output_ids.shape[1]
            break  # Only do one iteration for warmup
    except Exception as e:
        print(f"Warmup失败，跳过: {e}")
        pass


def build_prompt(tokenizer, model_type, history, system_msg=None):
    """Build chat prompt based on model_type and history."""
    if model_type == "llama-3-instruct":
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        for q, a in history:
            messages.append({"role": "user", "content": q})
            if a is not None:
                messages.append({"role": "assistant", "content": a})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    # fastchat-style templates for vicuna/llama-2-chat/mixtral
    if get_conversation_template is None:
        raise RuntimeError("fastchat is required for non-llama-3-instruct templates.")

    conv = get_conversation_template(model_type if model_type != "mixtral" else "llama-2-chat")
    if model_type == "llama-2-chat" and system_msg:
        conv.system_message = system_msg
    if model_type == "mixtral":
        conv.system_message = ""
        conv.sep2 = "</s>"

    for q, a in history:
        conv.append_message(conv.roles[0], q)
        if model_type == "llama-2-chat" and a:
            a = " " + a
        conv.append_message(conv.roles[1], a)

    prompt = conv.get_prompt()
    if model_type == "llama-2-chat":
        prompt += " "
    return prompt


def stream_generate(model, tokenizer, input_ids, use_eagle=True, temperature=0.0, top_p=1.0, max_new_tokens=512, is_llama3=False):
    """Yield incremental decoded text chunks during generation."""
    gen = (model.ea_generate if use_eagle else model.naive_generate)(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        is_llama3=is_llama3,
    )

    input_len = input_ids.shape[1]
    prev_len = input_len
    stop_token_ids = [tokenizer.eos_token_id]
    if is_llama3:
        try:
            stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        except Exception:
            pass

    for output_ids in gen:
        cur_len = output_ids.shape[1]
        if cur_len <= prev_len:
            continue
        # decode only the newly generated segment
        new_ids = output_ids[0, prev_len:cur_len].tolist()
        # Check for stop tokens in the full decoded sequence
        full_ids = output_ids[0, input_len:].tolist()
        stop_positions = [i for i, tid in enumerate(full_ids) if tid in stop_token_ids]
        if stop_positions:
            # truncate to first stop
            first_stop = stop_positions[0]
            full_ids = full_ids[:first_stop]
            # adjust new_ids accordingly
            if prev_len - input_len < len(full_ids):
                new_ids = full_ids[prev_len - input_len:]
            else:
                new_ids = []
        text_chunk = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        yield text_chunk, output_ids
        if stop_positions:
            break
        prev_len = cur_len



def main():
    parser = argparse.ArgumentParser(description="在终端中与LLM对话（支持EAGLE推测推理）")
    parser.add_argument("--ea-model-path", type=str, required=True, help="EAGLE草稿模型的路径（本地目录或HF仓库ID）")
    parser.add_argument("--base-model-path", type=str, required=True, help="基础LLM的路径（本地目录或HF仓库ID）")
    parser.add_argument("--model-type", type=str, default="llama-3-instruct", choices=["llama-2-chat", "vicuna", "mixtral", "llama-3-instruct"], help="聊天模板类型")
    parser.add_argument("--total-token", type=int, default=60, help="EAGLE-3的最大猜测token数（或-1使用默认）")
    parser.add_argument("--max-new-token", type=int, default=512, help="单轮生成的最大新token数")
    parser.add_argument("--temperature", type=float, default=0.5, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="采样Top-P")
    parser.add_argument("--load-in-8bit", action="store_true", help="使用8bit量化加载基础模型")
    parser.add_argument("--load-in-4bit", action="store_true", help="使用4bit量化加载基础模型")
    parser.add_argument("--no-eagle3", action="store_true", help="不使用EAGLE-3特性（降级到EAGLE-2实现）")
    parser.add_argument("--naive", action="store_true", help="使用基础模型的朴素自回归解码（不使用EAGLE加速）")
    args = parser.parse_args()

    # 构建模型
    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device_map="auto",
        use_eagle3=(not args.no_eagle3),
    )
    model.eval()
    
    # Warmup the model like webui.py does
    warmup(model, args)

    tokenizer = model.get_tokenizer()

    # 简单的系统提示（保持与webui一致）
    system_msg = None
    if args.model_type == "llama-2-chat":
        system_msg = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
            "If you don't know the answer to a question, please don't share false information."
        )

    history = []  # list of [user, assistant]

    print("已加载模型。现在可以开始对话（输入 exit 退出）。")
    while True:
        try:
            user_input = input("你: ")
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.strip().lower() in {"exit", "quit", ":q"}:
            print("退出。")
            break

        # 组装对话上下文
        history.append([user_input, None])
        prompt = build_prompt(tokenizer, args.model_type, history, system_msg)

        input_ids = tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).to(model.base_model.device)

        # 逐步流式打印输出
        print("助理: ", end="", flush=True)
        all_text = ""
        start_time = time.time()
        for chunk, output_ids in stream_generate(
            model,
            tokenizer,
            input_ids,
            use_eagle=(not args.naive),
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_token,
            is_llama3=(args.model_type == "llama-3-instruct"),
        ):
            if chunk:
                all_text += chunk
                sys.stdout.write(chunk)
                sys.stdout.flush()
        print()  # 换行
        # 将助理回复写入历史
        history[-1][1] = all_text

        # 打印本轮速度信息（可选）
        gen_tokens = len(output_ids[0]) - input_ids.shape[1]
        elapsed = time.time() - start_time
        if elapsed > 0:
            print(f"速度: {gen_tokens/elapsed:.2f} tokens/s")


if __name__ == "__main__":
    main()