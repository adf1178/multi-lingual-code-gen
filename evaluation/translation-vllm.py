import json
from vllm import LLM, SamplingParams

def translate(comment_list, llm, sampling_params):
    instruction = "Please translate the following English comment into Chinese:\n"
    
    prompts = [f"{instruction}###English comment: {comment}\nChinese comment:\n" for comment in comment_list]
    
    results = []
    outputs = llm.generate(prompts, sampling_params)
    

    for output in outputs:
        text = output.outputs[0].text
        results.append(text)
    return results



        
        
def main():
    #model_path = '/apdcephfs_cq2/share_1534723/models/checkpoint-600/'
    #model_path = '/dev/shm/codellama-13b-instruct'
    #model_path = '/dev/shm/checkpoint-800'
    # model_path = '/home/share/temp/tmp_czwang/cl-34b/checkpoint-400'
    model_path = '/home/share/czwang/models/codellama-34b/'
    #model_path = '/home/czwang/santacoder/checkpoints/checkpoint-1200/'    # 模型路径
    #model_path = '/apdcephfs_cq2/share_1534723/models/code-llama-python-7b/'
    #model_path = '/apdcephfs/private_sakurapeng/models/santacoder'
    #model_path='/dev/shm/codellama-7b-instruct'
    #model_path = '/apdcephfs/private_sakurapeng/models/deepseek-coder-7b-instruct'
    #model_path = '/home/czwang/models/santacoder-1b'
    #model_path = '/data/checkpoint-4800'
    #model_path = '/apdcephfs/private_sakurapeng/chat_codellama_34B/ckpt/codallama-34B-checkpoint-4800'
    # tokenizer_path = '/home/share/czwang/models/codellama-34b/'
    # tokenizer_path = '/home/share/temp/tmp_czwang/cl-34b'
    tokenizer_path = model_path
    parallel_size = 2     # gpu数量
    eval_path = 'human-eval-chinese-modified-dict_baiduTrans.json'
    #eval_path = 'human-eval-v2.json'    # 数据集
    batch_size = 164     # batch大小
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=128)
    llm = LLM(model=model_path, tensor_parallel_size=parallel_size, tokenizer=tokenizer_path, max_model_len=8192)
    print('load model success')
    with open("new.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i in range(0,164,30):
        chi_comments = [item['chinese_comment'] for item in data[i:i+30]]
    
    
    for i, item in enumerate(data):
        
        eng_comment = translate(item['chinese_comment'], llm, sampling_params)
        
        for idx, single_eng in enumerate(eng_comment):
            chinese_comment = item['chinese_comment'][idx].replace("\"\"\"", "").replace("#", "").strip()
            eng_comment = single_eng
            data[i]['prompt'] = data[i]['prompt'].replace(chinese_comment, eng_comment)
    new_dict = change(data)
    with open("save_ds_33b.json", 'w', encoding='utf-8') as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=4)
    print('done')

def change(data):
    # with open("save_ds_33b.json", 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    new_dict = {}
    for item in data:
        new_dict[item['task_id']] = item
    return new_dict
    # with open("save_ds_33b_base.json", 'w', encoding='utf-8') as f:
    #     json.dump(new_dict, f, ensure_ascii=False, indent=4)

main()