#model_path='/home/czwang/models/starcoder-gptq-8bit-128-act'
model_path='/home/share/czwang/models/codellama-34b'
# model_path='/mnt/cephfs/sakura/models/st_15b_other/checkpoint-1000'
token_path='/home/share/czwang/models/codellama-34b'
#model_path='/apdcephfs/private_sakurapeng/models/codegen25-7b'
input='humaneval_cpp.jsonl'
output='./multi-l-eval/cpp/cl-34b-base-eng.jsonl'
#output='./test-samples/cpp/starcoder-gptq-8bit-128-act.jsonl'
#python eval-x.py --model_path $model_path --output $output --batch_size 16  --resume
python3 vllm-x-eval.py --model_path $model_path --token_path $token_path --output $output --input $input --gpus 2