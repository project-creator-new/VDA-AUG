import json
import tqdm
import os
import random
import openai
from datetime import datetime
import argparse
import time


def make_requests(
        engine, prompts, max_tokens, temperature, top_p,
        system_ins, retries=3
):
    response = None
    target_length = max_tokens
    retry_cnt = 0
    backoff_time = 30
    while retry_cnt <= retries:
        try:
            client = openai.OpenAI(
                api_key="ffaa1b61872c9b2a3cc840d09cd51af5.JGTPyirMWSHaYfUf",
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )
            response = client.chat.completions.create(
                model=engine,
                messages=[
                    {"role": "system", "content": system_ins},
                    {"role": "user", "content": str(prompts)}
                ],
                top_p=top_p,
                temperature=temperature,
                max_tokens=target_length,
            )
            break
        except openai.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retry_cnt += 1
    data = {
        "prompt": prompts,
        "response": response.choices[0].message,
        "created_at": str(datetime.now()),
    }
    print(data['response'].content)
    print("*************************")
    return [data]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to GPT3.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from GPT3.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="The openai GPT3 engine to use.",
    )
    parser.add_argument(
        "--max_tokens",
        default=500,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of GPT3.",
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists."
    )
    parser.add_argument(
        "--request_batch_size",
        default=20,
        type=int,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # # 修改接口请求地址
    # openai.api_base = "https://api.f2gpt.com/v1"

    random.seed(123)
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # read existing file if it exists
    existing_responses = {}
    if os.path.exists(args.output_file) and args.use_existing_responses:
        with open(args.output_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                existing_responses[data["prompt"]] = data

    # do new prompts
    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = [json.loads(line)["prompt"] for line in fin]
        else:
            all_prompt = [line.strip().replace("\\n", "\n") for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
            batch_prompts = all_prompts[i: i + args.request_batch_size]
            if all(p in existing_responses for p in batch_prompts):
                for p in batch_prompts:
                    fout.write(json.dumps(existing_responses[p]) + "\n")
            else:
                results = make_requests(
                    engine=args.engine,
                    prompts=batch_prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop_sequences=args.stop_sequences
                )
                for data in results:
                    fout.write(json.dumps(data) + "\n")
