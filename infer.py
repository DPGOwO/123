import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
logging.getLogger("vllm").setLevel(logging.ERROR)

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--max_images_per_prompt', type=int, default=6)  # nuScenes多视角可设6
    return parser.parse_args()

def ensure_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def main():
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    # model_name = "Qwen3-VL-32B-Instruct"
    # task_name = "height"
    # qa_path = f"/home/ma-user/work/wx1427092/mydataset/key_objects_{task_name}QA_val.json"

    # image_basepath = "/cache/wx1427092/nuScenes/"
    # output_path = f"/home/ma-user/work/wx1427092/Qwen_output/{model_name}_{task_name}_{args.rank}.json"
    # model_path = f"/cache/wx1427092/{model_name}"

    exp_idx = 1
    model_name = "prompt_val23"
    qa_path = f"/cache/wx1427092/experiment/8/val_1.json"
    # qa_path = "/cache/wx1427092/experiment/8/vqa_plan_train_1.json"
    image_basepath = "/cache/wx1427092/nuscenes"
    # image_basepath = "/cache/wx1427092"
    # output_path = f"/cache/wx1427092/experiment/7/{model_name}_{exp_idx}_{args.rank}.json"
    output_path = f"/cache/wx1427092/experiment/8/{model_name}_{args.rank}.json"
    model_path = f"/cache/wx1427092/{model_name}"

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    # vLLM 引擎：continuous batching + 高效 KV 管理
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=16384 * 2,
        max_num_seqs=args.batch_size * 2,  # 让引擎更容易吃满
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
        # enforce_eager=True,  # 如遇到奇怪的 kernel/兼容性问题可打开（会慢一点）
    )

    sampling_params = SamplingParams(
        temperature=0.0,          # greedy
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=False,
    )

    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = data[args.rank::args.world_size]

    # ✅ 追加写，避免每个 batch 重写整个 json
    result = []
    for idx in tqdm(range(0, len(processed_data), args.batch_size),
                    position=args.rank,
                    leave=True,
                    dynamic_ncols=True,
                    desc=f"rank={args.rank}"):
        batch = processed_data[idx: idx + args.batch_size]
        if not batch:
            break

        llm_inputs = []
        meta = []  # 用来对齐输出

        for d in batch:
            question = d["conversations"][0]["value"]
            answer = d["conversations"][1]["value"]

            # d["image"] 既可能是 str，也可能是 list[str]
            img_list = ensure_list(d["image"])
            img_paths = [os.path.join(image_basepath, p) for p in img_list]

            content = [{"type": "image", "image": p} for p in img_paths]
            content.append({"type": "text", "text": question})
            
            messages = [{"role": "user", "content": content}]

            # ✅ 关键：生成纯 prompt（不要 tokenize）
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # ✅ 把 messages 里的 image path 解析成图像输入（供 vLLM multi_modal_data 使用）
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
            })

            meta.append((question, d["image"], answer, d["sample_token"]))

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

        for (q, img_field, ans, sample_token), out in zip(meta, outputs):
            gen_text = out.outputs[0].text  # vLLM 返回结构
            record = {
                "question": q,
                "image": img_field,
                "answer": ans,
                "output": gen_text,
                "sample_token": sample_token
            }
            result.append(record)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(f"Done. Wrote: {output_path}")

if __name__ == "__main__":
    main()
