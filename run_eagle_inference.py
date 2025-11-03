#!/usr/bin/env python3
"""
EAGLE推测性推理示例脚本
使用已训练的草稿模型进行推测性解码
"""

import torch
import os
import time
from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor

def main():
    # 模型路径配置 - 使用官方预训练模型
    # 基础模型：LLaMA 3.1-8B-Instruct
    base_model_path = "/home/jinjm/dev/models/llama31-8b-chat/LLM-Research/Meta-Llama-3___1-8B-Instruct"
    # 草稿模型：使用你训练的SpecForge模型
    ea_model_path = "/home/jinjm/dev/SpecForge/outputs/llama3-8b-eagle3-1000sample/epoch_1"
    
    print("正在加载EAGLE模型...")
    print(f"基础模型路径: {base_model_path}")
    print(f"草稿模型路径: {ea_model_path}")
    
    # 检查基础模型文件是否存在
    if not os.path.exists(base_model_path):
        print(f"错误: 基础模型路径不存在: {base_model_path}")
        return
    
    # 草稿模型使用HuggingFace路径，会自动下载，无需检查本地路径
    print(f"注意: 草稿模型将从HuggingFace自动下载: {ea_model_path}")
    
    # 加载EAGLE模型 - 使用官方评估脚本的参数
    try:
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            total_token=60,   # 官方默认值
            depth=5,          # 树深度
            top_k=10,         # top-k采样
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_eagle3=True,  # 使用EAGLE-3
        )
        print("✓ 模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 获取tokenizer
    tokenizer = model.get_tokenizer()
    model.eval()
    
    # 准备测试对话
    test_messages = [
        {
            "role": "system",
            "content": "你是一个有用的AI助手。请用中文回答问题。"
        },
        {
            "role": "user", 
            "content": "请介绍一下人工智能的发展历史，包括主要的里程碑事件。"
        }
    ]
    
    # 应用聊天模板
    prompt = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    print("\n=== 开始推测性推理 ===")
    print(f"输入提示: {test_messages[1]['content']}")
    print("\n生成的回答:")
    
    # 编码输入
    input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
    input_tensor = torch.as_tensor(input_ids).cuda()
    
    # 设置推理参数
    temperature = 0.7
    max_new_tokens = 512
    
    # 准备logits处理器
    logits_processor = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None
    
    # 记录开始时间
    torch.cuda.synchronize()
    start_time = time.time()
    
    # 执行EAGLE推测性推理
    try:
        output_ids, new_token, idx = model.eagenerate(
            input_tensor,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            log=True,
            is_llama3=True,
        )
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # 解码输出
        output_ids = output_ids[0][len(input_ids[0]):]
        
        # 处理停止token
        stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        if stop_token_ids:
            stop_token_ids_index = [
                i for i, id in enumerate(output_ids)
                if id in stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[:stop_token_ids_index[0]]
        
        # 解码文本
        output_text = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        
        # 清理特殊token
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output_text = output_text.replace(special_tok, "")
            else:
                output_text = output_text.replace(special_token, "")
        
        output_text = output_text.strip()
        
        print(output_text)
        print("\n=== 推理统计信息 ===")
        print(f"生成token数量: {new_token}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"生成速度: {new_token/total_time:.2f} tokens/秒")
        print(f"推测性推理步数: {idx}")
        
        # 计算加速比（与普通推理对比）
        print("\n=== 与普通推理对比 ===")
        torch.cuda.synchronize()
        naive_start = time.time()
        
        naive_output_ids, naive_new_token, naive_idx = model.naive_generate(
            input_tensor,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            log=True,
            is_llama3=True,
        )
        
        torch.cuda.synchronize()
        naive_time = time.time() - naive_start
        
        speedup = naive_time / total_time
        print(f"普通推理耗时: {naive_time:.2f}秒")
        print(f"普通推理速度: {naive_new_token/naive_time:.2f} tokens/秒")
        print(f"EAGLE加速比: {speedup:.2f}x")
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("EAGLE推测性推理演示")
    print("=" * 50)
    main()