

## Example of Java
Base models may struggle to determine where to stop generation, and this problem becomes more severe in Chinese instructions, here is an example:

```java
import java.util.*;
import java.lang.*;

class Solution {
    /*sum_to_n是一个函数，用于计算从1到n的所有数字的和。    >>> sum_to_n(30)
    465
    >>> sum_to_n(100)
    5050
    >>> sum_to_n(5)
    15
    >>> sum_to_n(10)
    55
    >>> sum_to_n(1)
    1
    */
    public int sumToN(int n) {
        return (n * (n + 1)) / 2;
    }

    /*sum_to_n_naive是一个函数，用于计算从1到n的所有数字的和。    >>> sum_to_n_naive(30)
    465
    >>> sum_to_n_naive(100)
    5050
    >>> sum_to_n_naive(5)
    15
    >>> sum_to_n_naive(10)
    55
    >>> sum_to_n_naive(1)
    1
    */
    public int sumToN_naive(int n) {
        int sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += i;
        }
        return sum;
    }

    /*sum_to_n_fast是一个函数，用于计算从1到n的所有数字的和。    >>> sum_to_n_fast(30)
    465
    >>> sum_to_n_fast(100)
    5050
    >>> sum_to_n_fast(5)
    15
    >>> sum_to_n_fast(10)
    55
    >>> sum_to_n_fast(1)
    1
    */
    public int sumToN_fast(int n) {
        return (n * (n + 1)) / 2;
    }

    /*sum_to_n_fastest是一个函数，用于计算从1到n的所有数字的和。    >>> sum_to_n_fastest(30)
    465
    >>> sum_to_n_fastest(100)
    5050
    >>> sum_to_n_fastest(5)
    15
    >>> sum_to_n_fastest(10)
    55
    >>> sum_to_n_fastest(1)
    1
    */
    public int sumToN_fastest(int n) {
        return n * (n + 1) / 2;
    }

    /*sum_to_n_fastestest是一个函数，用于计算从1到n的所有数字的和。    >>> sum_to_n_fastestest(30)
    465
    >>> sum_to_n_fastestest
```



# Experiment setup
## Data Dir
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
To use different deepspeed strategy, modify the line 295 of `training/chat-star.py`
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

# Details of Translation-Aware Tuning

We use the following template for translation aware tuning

```
Given an instruction in English, please first translate it into Chinese and then answer according to the instruction.

The English instruction is:
[ENG_INS]
The translated Chinese instruction is:
[CHI_INS]
Your answer is:
[CHI_ANS]

```

$[\mathit{ENG\_INS}]$, $[\mathit{CHI\_INS}]$, $[\mathit{CHI\_ANS}]$ represent English instruction, Chinese instruction, and corresponding answers in Chinese, respectively.