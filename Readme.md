# Experiment setup
##
Benchmark dir: `evaluation/benchmark`

Training data dir: `training/data`
## Evaluating

```bash
pip install vllm==0.2.1.post1
pip install evalplus
```

## How to Train

```bash
pip install deepspeed
conda install mpi4py
pip install torch
pip install transformers >= 4.34.1
```
Modify the bash file: training/mix-torch.sh

set nproc_per_node as your available nums of GPU

set --model_path as your model path

set --dataset_name as the path of the data

set --output_dir as the ckpt path
```bash
cd training
CUDA_VISIBLE_DEVICES=XXX bash mix-torch.sh
```

## How to Evaluate

### For Python
Modify evaluation/vllm-eval.py

Set the model path and output file name

If you want to evaluate Chinese, comment line 63. Instead, not comment this.

Then run `python vllm-eval.py`

After generating, run `evalplus.evaluate --dataset humaneval --samples XXX.jsonl`
### For C++ and Java

Modify evaluation/run_x.sh

Set the model path and output file name

Then run `bash run_x.sh`

After generating: go to the [HumanEval-X](https://github.com/THUDM/CodeGeeX/blob/main/codegeex/benchmark/README.md) project for computing pass@1.



# Details of EvolInstruct

We use the prompt in the [WizardCoder](https://arxiv.org/pdf/2306.08568.pdf?trk=public_post_comment-text)

Specifically, we use the following prompt:

```
Please increase the difficulty of the given programming test question a bit.

You can increase the difficulty using, but not limited to, the following methods:
{method}

{question}
```

The {method} includes

```
Add new constraints and requirements to the original problem, adding approximately 10 additional words.

Replace a commonly used requirement in the programming task with a less common and more specific one.

If the original problem can be solved with only a few logical steps, please add more reasoning steps.

Provide a piece of erroneous code as a reference to increase misdirection.

Propose higher time or space complexity requirements, but please refrain from doing so frequently.

```