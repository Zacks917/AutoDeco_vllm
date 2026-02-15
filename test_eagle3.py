from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
# CUDA_VISIBLE_DEVICES=1 python test_eagle3.py --mtp_size 1 --spe_model_path ../../models/RedHatAI/Qwen3-8B-speculator.eagle3
def main(args):
    import json
    import os
    from vllm import SamplingParams, LLM
    from transformers import AutoTokenizer

    # spe_model_path = "../../models/RedHatAI/Qwen3-8B-speculator.eagle3"
    # spec_model_path = "../../AutoMTP/V6"
    model_path = "/aifs4su/guhao/Models/Qwen3-8B-Base-RL"
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    if args.disable_should_stop:
        os.environ["DISABLE_SHOULD_STOP"] = "1"
    else:
        os.environ.pop("DISABLE_SHOULD_STOP", None)

    # Optional: synchronize should_stop across the whole batch.
    if getattr(args, "sync_stop_in_batch", False):
        os.environ["SYNC_SHOULD_STOP_IN_BATCH"] = "1"
    else:
        os.environ.pop("SYNC_SHOULD_STOP_IN_BATCH", None)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        disable_log_stats=False,
        enforce_eager=not args.use_cuda_graph,
        speculative_config={
            "model": f"{args.spe_model_path}",
            "num_speculative_tokens": args.mtp_size,
            "method": "eagle3"
        })
    # llm = LLM(model="../../models/Qwen/Qwen3-8B", tensor_parallel_size=1, disable_log_stats=False, speculative_config={"model": "../../AutoMTP/V6", "num_speculative_tokens": args.mtp_size, "method": "eagle3"})

    # with open("../data/qwen3_deepmath_generation_eval.jsonl", "r") as f:
    #     data = [json.loads(line) for line in f.readlines()]
    # # data = data[:20]
    # # debug

    # messages = [
    #     {"role": "user", "content": "Hello, world! Who are you? and how can you help me?"}
    # # ]
    # messages = [data[0]['messages'][0]]
    # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # prompts = [prompt]
    # 提取model_path倒数第三部分作为文件名

    # outputs = llm.generate([prompt], SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8192))[0]
    # print(outputs.outputs[0].text)
    # print(outputs.mtp_predictions)
    # assert False
    
    # # ========== Warmup (预热) ==========
    # print("=" * 50)
    # print("Warming up...")
    # warmup_prompt = prompts[0] if prompts else "Hello, world!"
    # for _ in range(3):  # 预热 3 次
    #     _ = llm.generate([warmup_prompt], SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100))
    # print("Warmup done!")
    # print("=" * 50)
    
    # # 清空输出文件
    # with open(output_file, "w") as f:
    #     pass
    
    prompts = []
    if args.data != "math500":
        raise ValueError(f"Invalid data: {args.data}, only 'math500' is supported.")

    should_stop_compute_only = os.getenv("SHOULD_STOP_COMPUTE_ONLY", "0") in ("1", "true", "True")
    sync_stop_in_batch = os.getenv("SYNC_SHOULD_STOP_IN_BATCH", "0") in ("1", "true", "True")
    output_file = (
        f'/aifs4su/guhao/MTP/logs/generation_log_new/'
        f'_disable_should_stop_{args.disable_should_stop}'
        f'_compute_only_{should_stop_compute_only}'
        f'_mtp_size_{args.mtp_size}.jsonl'
    )
    data_path = 'HuggingFaceH4/MATH-500'
    dataset = load_dataset(data_path, split='test')

    # 只取前 200 条样本
    dataset = dataset.select(range(256))
    for item in dataset:
        # MATH-500 中题目文本在 'problem' 字段里，这里转换为对话格式
        messages = [{"role": "user", "content": item["problem"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(prompt)
    print(f"Loaded {len(prompts)} prompts from {data_path}")

    # Simulation mode: repeat a single prompt to fill the whole batch.
    if getattr(args, "single_prompt_batch", False):
        if len(prompts) == 0:
            raise RuntimeError("No prompts loaded for single_prompt_batch mode.")
        prompts = [prompts[0]] * max(1, args.batch_size)
        print(f"[single_prompt_batch] Using 1 prompt repeated to batch_size={args.batch_size}")
        
    # ========== Batch 推理 (使用 vLLM 内部精确计时) ==========
    total_ac = 0
    total_re = 0
    total_output_length = 0
    mtp_step_size = []
    
    # 先清空文件（每次运行覆盖之前的结果）
    total_e2e_time = 0.0
    # with open(output_file, "w") as f:

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8192)
    bs = max(1, args.batch_size)
    total_batches = (len(prompts) + bs - 1) // bs
    global_idx = 0

    for batch_idx in range(total_batches):
        start = batch_idx * bs
        end = min((batch_idx + 1) * bs, len(prompts))
        batch_prompts = prompts[start:end]

        outputs = llm.generate(batch_prompts, sampling_params)
        print(f"[Batch {batch_idx + 1}/{total_batches}] size={len(batch_prompts)}")

        for output in outputs:
            global_idx += 1
            # 使用 vLLM 内部的精确时间统计
            metrics = output.metrics
            if metrics and metrics.finished_time and metrics.arrival_time:
                e2e_time = metrics.finished_time - metrics.arrival_time
            else:
                e2e_time = 0
                print(f"Warning: metrics not available for prompt {global_idx}")

            num_tokens = len(output.outputs[0].token_ids)
            throughput = num_tokens / e2e_time if e2e_time > 0 else 0
            total_output_length += num_tokens
            total_e2e_time += e2e_time
            print(f"  [{global_idx}/{len(prompts)}] E2E: {e2e_time:.4f}s, Tokens: {num_tokens}, Throughput: {throughput:.2f} tokens/s")

            # NOTE: Do not compute Ac/Re from mtp_predictions in batch mode.
            # Use scheduler-level spec_decoding_stats instead (see below).
            pass
            # print(f"Ac/All: {total_ac / total_output_length}, Re/All: {total_re / total_output_length}, Total Ac: {total_ac}, Total Re: {total_re}, Total Proposed: {total_ac+total_re}, Total Acc Ratio: {total_ac / (total_ac + total_re)}, Mean Output Length: {total_output_length / len(prompts)}, Mean MTP Step Size: {sum(mtp_step_size) / len(mtp_step_size)}")

            # assert False

            # # 每处理完一个 prompt 就追加写入
            # f.write(json.dumps({
            #     "prompt": prompt,
            #     "response": response,
            #     "mtp_predictions": mtp_predictions,
            #     "very_count": very_count,
            #     "output_count": output_count,
            #     "time_cost": e2e_time,
            #     "re_count": re_count
            # }) + "\n")
            # f.flush()
    
    output_file = output_file.replace(
        ".jsonl",
        f"_batch_size_{args.batch_size}.txt",
    )
    # ===== Spec decoding stats (scheduler-level, aggregated) =====
    # These counters are incremented from SchedulerStats.spec_decoding_stats.
    # They are robust under batch + ragged drafting, unlike mtp_predictions.
    spec_draft_tokens = None
    spec_accepted_tokens = None
    spec_num_drafts = None
    try:
        logger_manager = getattr(getattr(llm, "llm_engine", None),
                                 "logger_manager", None)
        if logger_manager is not None:
            prom_logger = getattr(logger_manager, "prometheus_logger", None)
            if prom_logger is not None:
                spec_prom = getattr(prom_logger, "spec_decoding_prom", None)
                if spec_prom is not None and getattr(
                        spec_prom, "spec_decoding_enabled", False):
                    # NOTE: prometheus_client internals: Counter._value.get()
                    spec_draft_tokens = int(
                        spec_prom.counter_spec_decode_num_draft_tokens[0]._value.get())  # type: ignore[attr-defined]
                    spec_accepted_tokens = int(
                        spec_prom.counter_spec_decode_num_accepted_tokens[0]._value.get())  # type: ignore[attr-defined]
                    spec_num_drafts = int(
                        spec_prom.counter_spec_decode_num_drafts[0]._value.get())  # type: ignore[attr-defined]
    except Exception as e:
        print(f"Warning: failed to read spec_decoding_stats counters: {e}")

    if spec_draft_tokens is None or spec_accepted_tokens is None:
        print("Warning: spec_decoding_stats not available; Ac/Re will be 0.")
        total_ac = 0
        total_re = 0
    else:
        total_ac = spec_accepted_tokens
        total_re = max(0, spec_draft_tokens - spec_accepted_tokens)

    total_throughput = total_output_length / total_e2e_time if total_e2e_time > 0 else 0
    total_acc_ratio = total_ac / (total_ac + total_re) if (total_ac + total_re) > 0 else 0
    mean_output_len = total_output_length / len(prompts) if len(prompts) > 0 else 0
    # A more meaningful "mean MTP step size" from scheduler stats:
    # avg_draft_tokens_per_draft = num_draft_tokens / num_drafts
    mean_mtp_step = (spec_draft_tokens / spec_num_drafts
                     if spec_draft_tokens is not None and spec_num_drafts
                     and spec_num_drafts > 0 else 0)
    ac_all = total_ac / total_output_length if total_output_length > 0 else 0
    re_all = total_re / total_output_length if total_output_length > 0 else 0
    with open(output_file, "w") as f:
        f.write(
            f"Batch Size: {args.batch_size}, "
            f"Ac/All: {ac_all}, Re/All: {re_all}, "
            f"Total Ac: {total_ac}, Total Re: {total_re}, "
            f"Total Proposed: {total_ac+total_re}, Total Acc Ratio: {total_acc_ratio}, "
            f"Mean Output Length: {mean_output_len}, Mean MTP Step Size: {mean_mtp_step}, "
            f"Total Throughput: {total_throughput} tokens/s"
        )

    print(
        f"Batch Size: {args.batch_size}, Ac/All: {ac_all}, Re/All: {re_all}, "
        f"Total Ac: {total_ac}, Total Re: {total_re}, Total Proposed: {total_ac+total_re}, "
        f"Total Acc Ratio: {total_acc_ratio}, Mean Output Length: {mean_output_len}, "
        f"Mean MTP Step Size: {mean_mtp_step}, Total Throughput: {total_throughput} tokens/s"
    )
    print(f"All results saved to: {output_file}")
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtp_size", type=int, default=4)
    parser.add_argument("--disable_should_stop", type=bool, default=False)
    parser.add_argument("--spe_model_path", type=str, default=None)
    parser.add_argument("--use_cuda_graph", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--single_prompt_batch", action="store_true", default=False)
    parser.add_argument("--sync_stop_in_batch", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="math500", choices=["math500"])
    args = parser.parse_args()
    mtp_size = args.mtp_size
    main(args)

