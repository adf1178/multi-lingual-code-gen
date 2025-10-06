# Code Language Model Training and Evaluation

A comprehensive framework for training and evaluating code language models with support for multiple programming languages and advanced instruction tuning methods.

---

## 📂 Project Structure

```
.
├── evaluation/
│   └── benchmark/          # Benchmark datasets
├── training/
│   └── data/              # Training datasets
├── evaluation/
│   ├── vllm-eval.py       # Python evaluation script
│   └── run_x.sh           # C++/Java evaluation script
└── training/
    ├── chat-star.py       # Training script
    └── mix-torch.sh       # Training launcher
```

---

## 🚀 Quick Start

### Prerequisites

**For Evaluation:**
```bash
pip install vllm==0.2.1.post1
pip install evalplus
```

**For Training:**
```bash
pip install deepspeed
conda install mpi4py
pip install torch
pip install transformers>=4.34.1
```

---

## 🏋️ Training

### Configuration

1. **Modify DeepSpeed Strategy** (Optional)
   - Edit line 295 in `training/chat-star.py` to select your desired DeepSpeed strategy

2. **Configure Training Parameters**
   - Edit `training/mix-torch.sh`:
     - `nproc_per_node`: Set to your available number of GPUs
     - `--model_path`: Path to your base model
     - `--dataset_name`: Path to your training data
     - `--output_dir`: Path to save checkpoints

### Launch Training

```bash
cd training
CUDA_VISIBLE_DEVICES=0,1,2,3 bash mix-torch.sh
```

---

## 🧪 Evaluation

### Python Evaluation

1. **Configure Evaluation**
   - Edit `evaluation/vllm-eval.py`:
     - Set model path
     - Set output file name
     - For Chinese evaluation: comment line 63
     - For English evaluation: uncomment line 63

2. **Generate Predictions**
   ```bash
   python vllm-eval.py
   ```

3. **Compute Metrics**
   ```bash
   evalplus.evaluate --dataset humaneval --samples XXX.jsonl
   ```

### C++ and Java Evaluation

1. **Configure Evaluation**
   - Edit `evaluation/run_x.sh`:
     - Set model path
     - Set output file name

2. **Generate Predictions**
   ```bash
   bash run_x.sh
   ```

3. **Compute Pass@1**
   - Follow instructions in the [HumanEval-X](https://github.com/THUDM/CodeGeeX/blob/main/codegeex/benchmark/README.md) project

---

## 📝 Methodology

### Evol-Instruct

We implement the Evol-Instruct method from [WizardCoder](https://arxiv.org/pdf/2306.08568.pdf) to automatically increase instruction complexity.

**Prompt Template:**
```
Please increase the difficulty of the given programming test question a bit.

You can increase the difficulty using, but not limited to, the following methods:
{method}

{question}
```

**Evolution Methods:**

- **Add Constraints**: Add new constraints and requirements to the original problem, adding approximately 10 additional words
- **Replace Requirements**: Replace a commonly used requirement with a less common and more specific one
- **Add Reasoning Steps**: If the problem can be solved with only a few logical steps, add more reasoning steps
- **Provide Erroneous Code**: Provide a piece of erroneous code as a reference to increase misdirection
- **Complexity Requirements**: Propose higher time or space complexity requirements (use sparingly)

### Translation-Aware Tuning

For multilingual support, we employ translation-aware tuning with the following template:

```
Given an instruction in English, please first translate it into Chinese and then answer according to the instruction.

The English instruction is:
[ENG_INS]
The translated Chinese instruction is:
[CHI_INS]
Your answer is:
[CHI_ANS]
```

Where:
- `[ENG_INS]`: English instruction
- `[CHI_INS]`: Chinese instruction
- `[CHI_ANS]`: Corresponding answer in Chinese

---

## 📌 Case Study

### Java Generation Challenge

Base models may struggle with proper stopping conditions, especially with Chinese instructions. Below is an example showing repetitive generation:

<details>
<summary>Click to view example</summary>

```java
import java.util.*;
import java.lang.*;

class Solution {
    /*sum_to_n是一个函数，用于计算从1到n的所有数字的和。
    >>> sum_to_n(30)
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

    /*sum_to_n_naive是一个函数，用于计算从1到n的所有数字的和。
    >>> sum_to_n_naive(30)
    465
    >>> sum_to_n_naive(100)
    5050
    ...
    */
    public int sumToN_naive(int n) {
        int sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += i;
        }
        return sum;
    }

    // Model continues generating similar methods unnecessarily...
    /*sum_to_n_fast是一个函数，用于计算从1到n的所有数字的和。*/
    public int sumToN_fast(int n) {
        return (n * (n + 1)) / 2;
    }

    /*sum_to_n_fastest是一个函数，用于计算从1到n的所有数字的和。*/
    public int sumToN_fastest(int n) {
        return n * (n + 1) / 2;
    }

    /*sum_to_n_fastestest是一个函数，用于计算从1到n的所有数字的和。
    >>> sum_to_n_fastestest(30)
    465
    >>> sum_to_n_fastestest
    // Generation continues without proper stopping...
```

</details>

This demonstrates the challenge of controlling generation length and stopping conditions, particularly in cross-lingual scenarios.

---

