# Main repo
This is the official implementation of the paper "Continual ToM: Benchmarking Belief Tracking in Language Models"
<br>
<br>

<!-- TABLE OF CONTENTS -->
### Table of contents
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#inference">Inference</a></li>
        <li><a href="#graphs">Graphs</a></li>
        <li><a href="#judge">Judge</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contacts">Contacts</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
  
<br>
<br>


## Getting Started

First of all clone the repo:
```sh
  git clone "url repo"
  cd RT_ToM
  ```

### Overview

Structure of the code:

```sh
  python main.py
```

<i>"graphs"</i> -> Single class analysis (so divided per sub-categories of a single benchmark) and custom graphs <br>
<i>"tables"</i> -> Cross benchmarks analysis (stats of the models on all benchmarks) <br>
<i>"judge_NER.py" and "judge_Beliefs.py"</i> -> Memory and beelief analysis (run first NER) <br>
<i>"run_HiToM0.py" and "run_ToMi_reality.py"</i> -> Reality table data (run models on HiToM 0th order and ToMI only reality questions). For HiToM change name on "evaluate_EbE.py" and "evaluate_all.py" to <i>_TABLE0</i> <br>
<i>"custom_dataset_process"</i> -> Run custom dataset evaluation. "compare_memories_custom_dataset.py" check the similarities of each predicted memory with the corrisponding ground truth (extracting with matching locations, first order beliefs and higher order beliefs). "judge_LLM_memory.py" compare the information present in th ememory predicted amd in the gold memory and give a score. <br>
<i>"VP_vs_EbE"</i> -> Run a comparison between the whole story (used in VP) and the last predicted memory before the question (used to answer in EbE), using a judge to give a similarity score for the information present. <br>

<br>
<br>


### Inference

To run the inference

```sh
  python main.py
```

With args: <br>
--method: Method to use, full story or event-by-event (default: story) <br>
--data: Benchamrk in which test the model (default: all) <br>
--run_type: Type of inference to run (default: '') <br>
--model: Model to evaluate (default: qwen2.5-7b) <br>
--max_samples: Max number of samples to evaluate (default: None -> all the samples) <br>
--batch_size: Batch size to use (default: 4) <br>
--engine: Engine to use for inference (default: vllm) <br>

<br>
<br>

### Graphs
To generate the graphs insert in the respective file your data (as shown inside the file), and then run:

```sh
  cd  graphs
  python bar_graphs.py (#to generate bar graphs)
  python graphs.py (#to generate point graphs)
  ```

<br>
<br>

### Judge
To run the judge (be sure you ran EbE inference before running this command, otherwise there aren't data to judge):

```sh
  python judge_memory.py
```
With args: <br>
--benchamrk: Benchamrk in which test the model (default: all) <br>
--judge: Model to use to judge the memory updates (default: qwen2.5-7b) <br>
--model: Model to evaluate (default: qwen2.5-7b) <br>
--batch_size: Batch size to use (default: 4) <br>
--engine: Engine to use for inference (default: vllm) <br>

<br>
<br>


## License
MIT License  
<br>
<br>

## Contacts
Matteo Zaramella - matteo.zaramella@vis.uni-stuttgart.de


<br>
<br>

