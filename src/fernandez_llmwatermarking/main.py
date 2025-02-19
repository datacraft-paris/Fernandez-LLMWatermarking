"""
Main script for watermark detection.
Test with:
    python -m wm_interactive.core.main --model_name smollm2-135m --prompt_path data/prompts.json --method maryland --delta 4.0 --ngram 1
"""

import os
import json
import time
import tqdm
import torch
import numpy as np
import pandas as pd
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from fernandez_llmwatermarking.generator import WmGenerator, OpenaiGenerator, MarylandGenerator
from fernandez_llmwatermarking.detector import WmDetector, OpenaiDetector, OpenaiDetectorZ, MarylandDetector, MarylandDetectorZ

# model names mapping
model_names = {
    # 'llama-3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'smollm2-135m': 'HuggingFaceTB/SmolLM2-135M-Instruct',
    'smollm2-360m': 'HuggingFaceTB/SmolLM2-360M-Instruct',   
}

CACHE_DIR = "/Users/datacraft/Fernandez-LLMWatermarking/static/hf_cache"


def load_prompts(json_path: str, prompt_type: str = "smollm", nsamples: int = None) -> list[dict]:
    """Load prompts from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        prompt_type: Type of prompt dataset (alpaca, smollm)
        nsamples: Number of samples to load (if None, load all)
    
    Returns:
        List of prompts
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File {json_path} not found")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if prompt_type == "alpaca":
        prompts = [{"instruction": item["instruction"]} for item in data]
    elif prompt_type == "smollm":
        prompts = []
        for item in data:
            prompt = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
            prompt += f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append({"instruction": prompt})
    else:
        raise ValueError(f"Prompt type {prompt_type} not supported")
    
    if nsamples is not None:
        prompts = prompts[:nsamples]
    
    return prompts 

def load_results(json_path: str, result_key: str = "result", nsamples: int = None) -> list[str]:
    """Load results from a JSONL file.
    
    Args:
        json_path: Path to the JSONL file
        result_key: Key to extract from each JSON line
        nsamples: Number of samples to load (if None, load all)
    
    Returns:
        List of results
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File {json_path} not found")
    
    results = []
    with open(json_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                results.append(data[result_key])
            if nsamples is not None and len(results) >= nsamples:
                break
    
    return results

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Name of the model to use. Choose from: llama-3.2-1b, smollm2-135m')

    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default=None,
                       help='Path to the prompt dataset. Required if --prompt is not provided')
    parser.add_argument('--prompt_type', type=str, default="smollm",
                       help='Type of prompt dataset. Only used if --prompt_path is provided')
    parser.add_argument('--prompt', type=str, nargs='+', default=None,
                       help='List of prompts to use. If not provided, prompts will be loaded from --prompt_path')

    # generation parameters
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature for sampling (higher = more random)')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top p for nucleus sampling (lower = more focused)')
    parser.add_argument('--max_gen_len', type=int, default=256,
                       help='Maximum length of generated text')

    # watermark parameters
    parser.add_argument('--method', type=str, default='none',
                       help='Watermarking method. Choose from: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.)')
    parser.add_argument('--method_detect', type=str, default='same',
                       help='Statistical test to detect watermark. Choose from: same (same as method), openai, openaiz, maryland, marylandz')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    parser.add_argument('--ngram', type=int, default=1,
                       help='n-gram size for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='For maryland method: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=2.0,
                       help='For maryland method: bias to add to greenlist tokens')
    parser.add_argument('--scoring_method', type=str, default='v2',
                       help='Method for scoring. Choose from: none (score every token), v1 (score when context unique), v2 (score when context+token unique)')

    # experiment parameters
    parser.add_argument('--nsamples', type=int, default=None,
                       help='Number of samples to generate from the prompt dataset')
    parser.add_argument('--do_eval', type=bool, default=True,
                       help='Whether to evaluate the generated text')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save results')

    return parser

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build model
    model_name = args.model_name.lower()
    if model_name not in model_names:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_names.keys())}")
    model_name = model_names[model_name]
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR
    ).to(device)

    # build watermark generator
    if args.method == "none":
        generator = WmGenerator(model, tokenizer)
    elif args.method == "openai":
        generator = OpenaiGenerator(model, tokenizer, args.ngram, args.seed)
    elif args.method == "maryland":
        generator = MarylandGenerator(model, tokenizer, args.ngram, args.seed, gamma=args.gamma, delta=args.delta)
    else:
        raise NotImplementedError("method {} not implemented".format(args.method))

    # load prompts
    if args.prompt is not None:
        prompts = args.prompt
        prompts = [{"instruction": prompt} for prompt in prompts]
    elif args.prompt_path is not None:
        prompts = load_prompts(json_path=args.prompt_path, prompt_type=args.prompt_type, nsamples=args.nsamples)
    else:
        raise ValueError("Either --prompt or --prompt_path must be provided")
    
    # (re)start experiment
    os.makedirs(args.output_dir, exist_ok=True)
    start_point = 0 # if resuming, start from the last line of the file
    if os.path.exists(os.path.join(args.output_dir, f"results.jsonl")):
        with open(os.path.join(args.output_dir, f"results.jsonl"), "r") as f:
            for _ in f:
                start_point += 1
    print(f"Starting from {start_point}")

    # generate
    all_times = []
    with open(os.path.join(args.output_dir, f"results.jsonl"), "a") as f:
        for ii in range(start_point, len(prompts)):
            # generate text
            time0 = time.time()
            prompt = prompts[ii]["instruction"]
            result = generator.generate(
                prompt, 
                max_gen_len=args.max_gen_len, 
                temperature=args.temperature, 
                top_p=args.top_p
            )
            time1 = time.time()
            # time chunk
            speed = 1 / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta)) 
            all_times.append(time1 - time0)
            print(f"Generated {ii:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            # log
            f.write(json.dumps({
                "prompt": prompt, 
                "result": result[len(prompt):],
                "speed": speed,
                "eta": eta}) + "\n")
            f.flush()
    print(f"Average time per prompt: {np.sum(all_times) / (len(prompts) - start_point) :.2f}")

    if args.method_detect == 'same':
        args.method_detect = args.method
    if (not args.do_eval) or (args.method_detect not in ["openai", "maryland", "marylandz", "openaiz"]):
        return
    
    # build watermark detector
    if args.method_detect == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed)
    elif args.method_detect == "openaiz":
        detector = OpenaiDetectorZ(tokenizer, args.ngram, args.seed)
    elif args.method_detect == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, gamma=args.gamma, delta=args.delta)
    elif args.method_detect == "marylandz":
        detector = MarylandDetectorZ(tokenizer, args.ngram, args.seed, gamma=args.gamma, delta=args.delta)

    # evaluate
    results = load_results(json_path=os.path.join(args.output_dir, f"results.jsonl"), result_key="result", nsamples=args.nsamples)
    log_stats = []
    with open(os.path.join(args.output_dir, 'scores.jsonl'), 'w') as f:
        for text in tqdm.tqdm(results):
            # get token details and pvalues
            token_details = detector.get_details(text, scoring_method=args.scoring_method)
            pvalues, aux_info = detector.get_pvalues_by_tok(token_details)
            # log stats
            log_stat = {
                'num_token': aux_info['ntoks_scored'],
                'score': aux_info['final_score'],
                'pvalue': aux_info['final_pvalue'],
                'log10_pvalue': np.log10(aux_info['final_pvalue']),
            }
            log_stats.append(log_stat)
            f.write('\n' + json.dumps({k: float(v) for k, v in log_stat.items()}))
        df = pd.DataFrame(log_stats)
        print(f">>> Scores: \n{df.describe(percentiles=[])}")
        print(f"Saved scores to {os.path.join(args.output_dir, 'scores.csv')}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)