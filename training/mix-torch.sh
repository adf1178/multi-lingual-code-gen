#!/bin/bash


# Force hf api run offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export GPU_NUM=`nvidia-smi -L | wc -l`
export OMP_NUM_THREADS=23  # Suanli ping each gpu with fixed num of cpu

# Extract pytorch distributed-params from Gemini environment variable
export MASTER_PORT=${MASTER_PORT:=34567}
export RANK=${RANK:=0}
export WORLD_SIZE=${WORLD_SIZE:=1}

# This line is necessary, Yard/H800-Seczone will start user application through a new 
# account.
#git lfs install

# First, download pretrained-model and datasets from https://hub.adams.woa.com/
#git clone "${GitUrl}/${PretrainedModel}.git" ./pretrained_model
#git clone "${GitUrl}/${Dataset}.git" ./dataset

# Then download ckpt from hdfs
#hadoop fs -mkdir -p ${CkptDir}
#hadoop fs -get ${CkptDir} ./ckpt

# Use pytorch distributed launcher, you can use deepspeed through `deepspeed` param in `TrainingArguments`
python3 -m torch.distributed.run \
  --nproc_per_node=4 \
  --nnode=1 \
  --node_rank=${RANK} \
  --master_addr=127.0.0.1 \
  --master_port=11451 \
  ./chat-star.py --model_path="~/models/codellama-34b/" \
  --dataset_name="~/LCMs/Eng_MEIC.json"  \
  --subset="data/finetune" \
  --seq_length 2048 \
  --num_train_epochs 10 \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --lr_scheduler_type="cosine" \
  --num_warmup_steps 100 \
  --weight_decay 0.05 \
  --output_dir="~/cl-34b" \

# Last, upload ckpt to HDFS
#hadoop fs -put ./ckpt ${CkptDir}


        # // "offload_param": {
        # //     "device": "cpu",
        # //     "pin_memory": true 
        # // },