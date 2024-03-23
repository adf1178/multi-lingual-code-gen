model_path='XXXX/models/codellama-34b'
token_path='XXXX/models/codellama-34b'
input='humaneval_cpp.jsonl'
output='./multi-l-eval/cpp/cl-34b-base-eng.jsonl'
python3 vllm-x-eval.py --model_path $model_path --token_path $token_path --output $output --input $input --gpus 2