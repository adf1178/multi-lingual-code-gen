import time
import json
from vllm import LLM, SamplingParams
import re
from evalplus.data import get_human_eval_plus, write_jsonl


def remove_after_return(code):
    """
    Takes as input a code, and removes everything that is after the return.
    That is, the first line that does not start with a space character
    """
    pattern = r"[^\n]+(\n|$)"
    end_last_match = None
    # Go trough the regex to match any sequence of characters ending with a \n
    for match in re.finditer(pattern, code):
        start_match, end_match = match.span()
        # Search for the first line which does not start by a space character
        if (
            end_last_match is not None
            and start_match < len(code)
            and code[start_match].strip() != ""
        ):
            return code[0:start_match]
        end_last_match = end_match
    return code

def remove_end(code):
    pattern = r"[^\n]+(\n|$)"
    last_match = None
    for match in re.finditer(pattern, code):
        start_match, end_match = match.span()
        if code[start_match:end_match].strip() != "":
            last_match = end_match
    if last_match is not None:
        return code[:last_match]
    return code

def remove_code(code):
    lines = code.split('\n')
    add = False
    change = False
    result = []
    for line in lines:
        if line.startswith('```'):
            if add:
                break
            else:
                add = True
                change = True
                continue
        if add:
            result.append(line)
    if not change:
        return code
    return '\n'.join(result)

def generate(llm, sampling_params, file_path, batch_size):
    problems = json.load(open(file_path, "r"))
    print(f'{file_path} total: {len(problems)}')
    start = time.time()
    num = 0
    # problems = get_human_eval_plus()
    fim = False
    instruction = ""
    #instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
    #instruction = "<s>[INST] Write a python function that adds two numbers [/INST]"
    # instruction = "Create a Python script for this problem"
    #instruction = "Complete the following code with functionality specifications\n"
    #instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    #instruction = "<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise\n"
    #instruction = "You are an expert programmer that writes simple, concise code and explanations\n"
    #instruction = "Provide answers in python"
   
    if fim:
        prompts = [instruction + problems[task_id]['prompt'].strip() + "<FILL_ME>" for task_id in problems]
        #prompts = ["<fim_prefix>" + problems[task_id]['prompt'].strip() + "<fim_suffix><fim_middle>" for task_id in problems]
    else:
        prompts = [instruction + problems[task_id]['prompt'].strip().replace("\n\n", "") for task_id in problems]
        
        #prompts = [f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{problems[task_id]['prompt'].strip()}\n\n### Response:\n{problems[task_id]['prompt'].strip()}" for task_id in problems]
        
    vllm_input = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    tasks = [task_id for task_id in problems]
    # print(len(tasks))
    input_token_num = 0
    output_token_num = 0
    i = 0
    result = []
    for item in vllm_input:
        infer_start = time.time()
        outputs = llm.generate(item, sampling_params)
        for _, output in enumerate(outputs):
            text = output.outputs[0].text
            input_token_num += len(output.prompt_token_ids)
            output_token_num += len(output.outputs[0].token_ids)
            text = remove_after_return(text)
            text = remove_code(text)
            text = remove_end(text)
            task = tasks[i]   
            # full_text = text 
            full_text = problems[task]['prompt'] + text
            result.append({'task_id': task, 'completion': [full_text]})
            i += 1
    write_jsonl("./multi-l-eval/cl-34b-base.jsonl", result)
    #write_jsonl("./samples_star.jsonl", result)
    print(f'total_cost_time: {time.time() - start}s')
    print(f"input_token_num: {input_token_num}")
    print(f"output_token_num: {output_token_num}")

def main():
    model_path = 'XXXX Model Path'

    tokenizer_path = model_path
    parallel_size = 2     # num of gpus
    # eval_path = 'human-eval-chinese-modified-dict_baiduTrans.json'
    eval_path = 'human-eval-chinese-modified-dict.json'    # dataset
    batch_size = 164     # batch
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=512)
    llm = LLM(model=model_path, tensor_parallel_size=parallel_size, tokenizer=tokenizer_path, max_model_len=8192)
    print('load model success')
    generate(llm, sampling_params, eval_path, 164)
    #generate(llm, sampling_params, eval_path, 20)
    #generate(llm, sampling_params, eval_path, 1)
    

if __name__ == '__main__':
    main()