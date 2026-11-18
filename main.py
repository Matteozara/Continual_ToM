import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_data import *
'''from evaluate_HiToM_all import eval_HiToM_all
from evaluate_BigToM_all import eval_BigToM_all
from evaluate_ToMi_all import eval_ToMi_all
from evaluate_ToMChallenges_all import eval_ToMC_all

from evaluate_HiToM_EbE import eval_HiToM_EbE
from evaluate_BigToM_EbE import eval_BigToM_EbE
from evaluate_ToMChallenges_EbE import eval_ToMC_EbE
from evaluate_ToMi_EbE import eval_ToMi_EbE'''
from evaluate_all import eval_tom_benchmarks
from evaluate_EbE import eval_tom_benchmarks_EbE

from utils import load_model, set_seed, MODELS_MAP, BENCHMARKS
from dotenv import load_dotenv
from huggingface_hub import login



load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in environment.")

PATH_HiToM = "./data/eval/HiToM/hitom.json"
PATH_BigToM = "./data/eval/BigToM/bigtom.json"
PATH_ToMi = "./data/eval/ToMi/"
PATH_ToMChallenges = "./data/eval/ToMChallenges/"

#EXPERTS_PATH = {"df": "dodell_feder", "k": "kosinski", "u": "ullman"}

#casperhansen/llama-3.3-70b-instruct-awq
#mistralai/Ministral-3-14B-Instruct-2512
#deepseek-ai/DeepSeek-R1-Distill-Llama-8B


#set_seed(42)







##################
#   Evaluation   #
##################


def start_evaluation(args):
    print("Setting seeds ...")
    set_seed(42)
    if args.engine == "vllm":
        print("Setting vllm ...")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        #os.environ["VLLM_BATCH_INVARIANT"] = "1"

    print("Loading model ...")
    tokenizer, model = load_model(MODELS_MAP[args.model], engine_type=args.engine)

    if args.data == "all":
        benchs = BENCHMARKS
        
    else:
        benchs = [args.data]
    
    method = args.method
    summary = {}

    for benchmark in benchs:
        print(f"\nEvaluating benchmark: {benchmark}")

        if benchmark == "HiToM":
            if method == "events":
                summary["HiToM_EbE"] = eval_tom_benchmarks_EbE(tokenizer, model, model_name=args.model, data_path=PATH_HiToM, benchmark=benchmark, max_samples=args.max_samples, batch_size=args.batch_size, engine_type=args.engine, ebe_type=args.ebe_type)
            else:
                summary["HiToM_All"] = eval_tom_benchmarks(tokenizer, model, model_name=args.model, data_path=PATH_HiToM, benchmark=benchmark, max_samples=args.max_samples, suffix=args.run_type, batch_size=args.batch_size, engine_type=args.engine, type_variation=args.type_variation) #eval_HiToM_all(tokenizer, model, model_name=args.model, data_path=PATH_HiToM, max_samples=args.max_samples, cot=args.cot, batch_size=batch_size)


        elif benchmark == "BigToM":
            if method == "events":
                summary["BigToM_EbE"] = eval_tom_benchmarks_EbE(tokenizer, model, model_name=args.model, data_path=PATH_BigToM, benchmark=benchmark, max_samples=args.max_samples, batch_size=args.batch_size, engine_type=args.engine, ebe_type=args.ebe_type)
                #summary["BigToM_EbE"] = eval_BigToM_EbE(tokenizer, model, model_name=args.model, data_path=PATH_BigToM, max_samples=args.max_samples)
            else:
                summary["BigToM_All"] = eval_tom_benchmarks(tokenizer, model, model_name=args.model, data_path=PATH_BigToM, benchmark=benchmark, max_samples=args.max_samples, suffix=args.run_type, batch_size=args.batch_size, engine_type=args.engine, type_variation=args.type_variation)
                #summary["BigToM_All"] = eval_BigToM_all(tokenizer, model, model_name=args.model, data_path=PATH_BigToM, max_samples=args.max_samples, cot=args.cot, batch_size=batch_size)

        elif benchmark == "experts":
            #TO IMPLEMENT OR NO?
            path = ("./data/eval/expert_data/" + EXPERTS_PATH[args.split_e] + ".json")
            loader = load_expert(path, args.split_e)

        elif benchmark == "ToMChallenges":
            if method == "events":
                summary["ToMChallenges_EbE"] = eval_tom_benchmarks_EbE(tokenizer, model, model_name=args.model, data_path=PATH_ToMChallenges, benchmark=benchmark, max_samples=args.max_samples, batch_size=args.batch_size, engine_type=args.engine, ebe_type=args.ebe_type)
                #summary["ToMChallenges_EbE"] = eval_ToMC_EbE(tokenizer, model, model_name=args.model, data_path=PATH_ToMChallenges, max_samples=args.max_samples)
            else:
                summary["ToMChallenges_All"] = eval_tom_benchmarks(tokenizer, model, model_name=args.model, data_path=PATH_ToMChallenges, benchmark=benchmark, max_samples=args.max_samples, suffix=args.run_type, batch_size=args.batch_size, engine_type=args.engine, type_variation=args.type_variation)
                #summary["ToMChallenges_All"] = eval_ToMC_all(tokenizer, model, model_name=args.model, data_path=PATH_ToMChallenges, max_samples=args.max_samples, cot=args.cot, batch_size=batch_size)

        elif benchmark == "ToMi":
            if method == "events":
                summary["ToMi_EbE"] = eval_tom_benchmarks_EbE(tokenizer, model, model_name=args.model, data_path=PATH_ToMi, benchmark=benchmark, max_samples=args.max_samples, batch_size=args.batch_size, engine_type=args.engine, ebe_type=args.ebe_type)
                #summary["ToMi_EbE"] = eval_ToMi_EbE(tokenizer, model, model_name=args.model, data_path=PATH_ToMi, max_samples=args.max_samples)
            else:
                summary["ToMi_All"] = eval_tom_benchmarks(tokenizer, model, model_name=args.model, data_path=PATH_ToMi, benchmark=benchmark, max_samples=args.max_samples, suffix=args.run_type, batch_size=args.batch_size, engine_type=args.engine, type_variation=args.type_variation)
                #summary["ToMi_All"] = eval_ToMi_all(tokenizer, model, model_name=args.model, data_path=PATH_ToMi, max_samples=args.max_samples, cot=args.cot, batch_size=batch_size)

        else:
            continue





        
    return summary



############
#   Main   #
############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", choices=["story", "events"], default="story", help="Method to use, full story or event-by-event (default: story)")
    parser.add_argument("--data", choices=BENCHMARKS + ["all"], default="all", help="Benchamrk in which test the model (default: all)")
    parser.add_argument("--run_type", choices=["cot", "complete", ""], default="", help="Type of inference to run (default: '')")
    parser.add_argument("--ebe_type", choices=["standard", "location", "belief", "obj_location", "task_specific", "memory", "variation1", "variation2", "variation3", "variation4", "variation5"], default="standard", help="Type of inference to run EbE (with ablation variants of the standard version) (default: 'standard')")
    parser.add_argument("--type_variation", choices=["vp_1", "vp_2", "cot_1", "cot_2", "standard"], default="standard", help="Type of inference to run EbE (with ablation variants of the standard version) (default: 'standard')")
    parser.add_argument("--model", choices=list(MODELS_MAP.keys()), default="qwen2.5-7b", help="Model to evaluate (default: qwen2.5-7b)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to evaluate (default: None -> all the samples)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to use (default: 4)")
    parser.add_argument("--engine", choices=["vllm", "hf"], default="vllm", help="Engine to use for inference (default: vllm)")
    
    args = parser.parse_args()

    start_evaluation(args)