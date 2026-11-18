from evaluate_CLOSED_SOURCE_batch import eval_tom_benchmarks
from evaluate_EbE_CLOSED_SOURCE_batch import eval_tom_benchmarks_EbE 
import asyncio

# ==========================
# Configuration
# ==========================

MODEL_NAME = "gpt-5.4" #"gpt-5.5" #"gpt-5.4" #"gpt-5.6-luna" #"gpt-5.1" #"gpt-5.4" #"gpt-5-nano"

# Choose one:

#BENCHMARK = "HiToM"
#DATA_PATH = "./data/eval/HiToM/hitom.json"

#BENCHMARK = "HiToM0"
#DATA_PATH = "./data/eval/HiToM/hitom0.json"


#BENCHMARK = "BigToM"
#DATA_PATH = "./data/eval/BigToM/bigtom.json"

#BENCHMARK = "ToMChallenges"
#DATA_PATH = "./data/eval/ToMChallenges/"

BENCHMARK = "ToMi"
DATA_PATH = "./data/eval/ToMi/"

#BENCHMARK = "ToMi_reality"
#DATA_PATH = "./data/eval/ToMi/ToMi_reality.json"





# Prompt variant:
# "", "cot", "complete" (standard evaluation)
PROMPT_TYPE = "cot"

# EbE memory variant:
# "standard", ...
EBE_TYPE = "standard" #"standard" #"obj_location" #"standard"

# Which evaluation to run:
# "standard" or "ebe"
EVAL_METHOD = "ebe"

# Evaluate the whole dataset (or set an integer)
MAX_SAMPLES = None

# Always use the OpenAI API
ENGINE_TYPE = "openai"


def main():
    print("=" * 60)
    print(f"Model:      {MODEL_NAME}")
    print(f"Benchmark:  {BENCHMARK}")
    print(f"Dataset:    {DATA_PATH}")
    print(f"Method:     {EVAL_METHOD}")

    if EVAL_METHOD == "standard":
        print(f"Prompt:     {PROMPT_TYPE or 'default'}")
    else:
        print(f"EbE type:   {EBE_TYPE}")

    print("=" * 60)

    if EVAL_METHOD == "standard":

        asyncio.run(
            eval_tom_benchmarks(
                tokenizer=None,
                model=None,
                model_name=MODEL_NAME,
                benchmark=BENCHMARK,
                data_path=DATA_PATH,
                max_samples=MAX_SAMPLES,
                suffix=PROMPT_TYPE,
                engine_type=ENGINE_TYPE,
            )
        )

        '''eval_tom_benchmarks(
            tokenizer=None,
            model=None,
            model_name=MODEL_NAME,
            benchmark=BENCHMARK,
            data_path=DATA_PATH,
            max_samples=MAX_SAMPLES,
            suffix=PROMPT_TYPE,
            engine_type=ENGINE_TYPE,
        )'''

    elif EVAL_METHOD == "ebe":

        asyncio.run(
            eval_tom_benchmarks_EbE(
                tokenizer=None,
                model=None,
                model_name=MODEL_NAME,
                benchmark=BENCHMARK,
                data_path=DATA_PATH,
                max_samples=MAX_SAMPLES,
                engine_type=ENGINE_TYPE,
                ebe_type=EBE_TYPE,
            )
        )


        '''
        eval_tom_benchmarks_EbE(
                tokenizer=None,
                model=None,
                model_name=MODEL_NAME,
                benchmark=BENCHMARK,
                data_path=DATA_PATH,
                max_samples=MAX_SAMPLES,
                engine_type=ENGINE_TYPE,
                ebe_type=EBE_TYPE,
        )
        '''

    else:
        raise ValueError(
            f"Unknown EVAL_METHOD: {EVAL_METHOD}"
        )

    print("Done!")


if __name__ == "__main__":
    main()