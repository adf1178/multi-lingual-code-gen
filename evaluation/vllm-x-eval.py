import time
import json
from vllm import LLM, SamplingParams
import re
import argparse


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
            code =  code[0:start_match]
        end_last_match = end_match
    if "public class Main" in code:
        code = code.split("public class Main")[0]
    return code

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

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

def remove_after_main(code):
    lines = code.split('\n')
    new_lines = []
    for line in lines:
        if line.startswith('int main') or line.startswith('public class') or line.startswith('\'\'\''):
            break
        new_lines.append(line)
    new_code = '\n'.join(new_lines)
    return new_code

def generate(llm, sampling_params, file_path, batch_size, args):
    json_list = read_jsonl(file_path)
    print(f'{file_path} total: {len(json_list)}')
    start = time.time()
    num = 0
    fim = args.fim
    instruction = ''
    #instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request"
    #instruction = "Create a C++ script for this problem:"
    #instruction = "Complete the following code with C++ syntax specifications"
    #instruction = "<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise\n"
    if 'codellama' in args.model_path or 'code-llama' in args.model_path:
        if fim:
            prompts = [instruction + item['prompt'].strip() + '<FILL_ME>' for item in json_list]
        else:
            prompts = [instruction + item['prompt'].strip() for item in json_list]
    
    elif 'starcoder' in args.model_path:
        if fim:
            prompts = ["<fim_prefix>" + item['prompt'].strip() + "<fim_suffix><fim_middle>" for item in json_list ]
        else:
            prompts = [instruction + item['prompt'].strip() for item in json_list]
            
    elif 'santacoder' in args.model_path:
        if fim:
            prompts = ["<fim-prefix>" + item['prompt'].strip() + "<fim-suffix><fim-middle>" for item in json_list ]
        else:
            prompts = [instruction + item['prompt'].strip() for item in json_list]
    
    else:
        prompts = [instruction + item['prompt'].strip() for item in json_list]
    # if fim:
    #     prompts = ["<fim-prefix>" + item['prompt'].strip() + "<fim-suffix><fim-middle>" for item in json_list ]
    # else:
    #     prompts = [instruction + item['prompt'].strip() for item in json_list]
    vllm_input = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
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
            text = remove_after_main(text)
            #text = remove_code(text)
            if fim:
                prompt = json_list[i]['prompt']
            else:
                prompt = output.prompt
            print(prompt + text + '\n\n')
            with open(args.output, 'a') as file:
                json_data = json.dumps({'task_id': json_list[i]['task_id'], 'generation': text, 'prompt': prompt})
                file.write(json_data + '\n')
            i += 1
    print(f'total_cost_time: {time.time() - start}s')
    print(f"input_token_num: {input_token_num}")
    print(f"output_token_num: {output_token_num}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--token_path", type=str, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--fim", action="store_true", default=False)
    args = parser.parse_args()
    
    #model_path = '/home/czwang/models/starcoder'    # 模型路径
    model_path = args.model_path
    #model_path = '/apdcephfs/private_sakurapeng/models/codellama-7b-python'
    tokenizer_path = args.token_path
    #tokenizer_path = model_path
    #model_path = '/dev/shm/tmp/checkpoint-800/'
    #tokenizer_path = '/home/czwang/models/santacoder-wxg'
    parallel_size = args.gpus     # gpu数量
    #eval_path = 'humaneval_java.jsonl'
    eval_path = args.input    # 数据集
    batch_size = 164     # batch大小
    sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=512)
    llm = LLM(model=model_path, tensor_parallel_size=parallel_size, trust_remote_code=True,tokenizer=tokenizer_path, max_model_len=8192)
    print('load model success')
    generate(llm, sampling_params, eval_path, 164, args)
    #generate(llm, sampling_params, eval_path, 20)
    #generate(llm, sampling_params, eval_path, 1)
    

if __name__ == '__main__':
    main()