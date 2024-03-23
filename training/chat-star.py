import argparse
import os
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, logging, set_seed
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

"""
Fine-Tune StarCoder on Code Alpaca/SE
"""


class SavePeftModelCallback(TrainerCallback):

  def on_save(
      self,
      args: TrainingArguments,
      state: TrainerState,
      control: TrainerControl,
      **kwargs,
  ):
    checkpoint_folder = os.path.join(
        args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

    kwargs["model"].save_pretrained(checkpoint_folder)

    pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
    torch.save({}, pytorch_model_path)
    return control





def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, default="bigcode/large-model")
  parser.add_argument("--dataset_name",
                      type=str,
                      default="HuggingFaceH4/CodeAlpaca_20K")
  parser.add_argument("--subset", type=str)
  parser.add_argument("--split", type=str)
  parser.add_argument("--size_valid_set", type=int, default=10000)
  parser.add_argument("--streaming", action="store_true")
  parser.add_argument("--shuffle_buffer", type=int, default=5000)

  parser.add_argument("--input_column_name", type=str, default="prompt")
  parser.add_argument("--output_column_name", type=str, default="completion")

  parser.add_argument("--seq_length", type=int, default=2048)
  parser.add_argument("--max_steps", type=int, default=1000)
  parser.add_argument("--num_epochs", type=int, default=3)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
  parser.add_argument("--eos_token_id", type=int, default=49152)

  parser.add_argument("--lora_r", type=int, default=16)
  parser.add_argument("--lora_alpha", type=int, default=32)
  parser.add_argument("--lora_dropout", type=float, default=0.05)

  parser.add_argument("--learning_rate", type=float, default=5e-6)
  parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
  parser.add_argument("--num_warmup_steps", type=int, default=100)
  parser.add_argument("--weight_decay", type=float, default=0.05)

  parser.add_argument("--local_rank", type=int, default=0)
  parser.add_argument("--num_train_epochs", type=int, default=3)
  parser.add_argument("--no_fp16", action="store_false")
  parser.add_argument("--bf16", action="store_true", default=True)
  parser.add_argument("--no_gradient_checkpointing",
                      action="store_false",
                      default=False)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num_workers", type=int, default=None)
  parser.add_argument("--output_dir", type=str, default="./checkpoints")
  parser.add_argument("--log_freq", default=10, type=int)
  parser.add_argument("--eval_freq", default=100, type=int)
  parser.add_argument("--save_freq", default=200, type=int)
  parser.add_argument("--resume_from_checkpoint", default=False, type=str)
  return parser.parse_args()


def chars_token_ratio(dataset,
                      tokenizer,
                      input_column_name="prompt",
                      output_column_name="completion",
                      nb_examples=400):
  """
    Estimate the average number of characters per token in the dataset.
  """
  total_characters, total_tokens = 0, 0
  for _, example in tqdm(zip(range(nb_examples), iter(dataset)),
                         total=nb_examples):
    text = prepare_sample_text(example, input_column_name, output_column_name)
    total_characters += len(text)
    if tokenizer.is_fast:
      total_tokens += len(tokenizer(text).tokens())
    else:
      total_tokens += len(tokenizer.tokenize(text))

  return total_characters / total_tokens


def print_trainable_parameters(model):
  """
    Prints the number of trainable parameters in the model.
    """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
  )


def prepare_sample_text(example):
  """Prepare the text from a sample of the dataset."""
  #text = example['instruction_chi'] + '\n' + example['output']
  #text = example['prompt']
  text = example['instruction'] + '\n' + example['output']
  return text


class ConstantLengthDataset(IterableDataset):
  """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

  def __init__(
      self,
      tokenizer,
      dataset,
      infinite=False,
      seq_length=1024,
      num_of_sequences=1024,
      chars_per_token=3.6,
      # input_column_name="prompt",
      # output_column_name="completion"
  ):
    self.tokenizer = tokenizer
    self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else args.eos_token_id
    self.dataset = dataset
    self.seq_length = seq_length
    self.infinite = infinite
    self.current_size = 0
    self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
    # self.output_column_name = output_column_name

  def __len__(self):
      return len(self.dataset)

  def __iter__(self):
    iterator = iter(self.dataset)
    more_examples = True
    while more_examples:
      buffer, buffer_len = [], 0
      while True:
        if buffer_len >= self.max_buffer_size:
          break
        try:
          buffer.append(prepare_sample_text(next(iterator)))
          buffer_len += len(buffer[-1])
        except StopIteration:
          if self.infinite:
            iterator = iter(self.dataset)
          else:
            more_examples = False
            break
      tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
      all_token_ids = []
      for tokenized_input in tokenized_inputs:
        #all_token_ids.extend(tokenized_input)
        all_token_ids.extend(tokenized_input + [self.concat_token_id])
      mask = True
      for i in range(0, len(all_token_ids), self.seq_length):
        input_ids = all_token_ids[i:i + self.seq_length]
        if len(input_ids) == self.seq_length:
          self.current_size += 1
          labels = input_ids.copy()
          if mask:
              mask_user_labels(self.tokenizer, labels)
          yield {
              "input_ids": torch.LongTensor(input_ids),
              "labels": torch.LongTensor(labels),
          }

def mask_user_labels(tokenizer, labels):
    """Masks the user turns of a dialogue from the loss"""
    user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    for idx, label_id in enumerate(labels):
        if label_id == user_token_id:
            print('mask')
            current_idx = idx
            while current_idx < len(labels) and labels[current_idx] != assistant_token_id:
                labels[current_idx] = -100
                current_idx += 1

def create_datasets(tokenizer, args):
  all_data = load_dataset(
      'json',
      data_files=args.dataset_name,
      split='train',
      cache_dir="/dev/shm",
      num_proc=args.num_workers if not args.streaming else None,
  )
  new_data = all_data.train_test_split(test_size=0.05, shuffle=True)
  train_data = new_data["train"]
  valid_data = new_data["test"]
  print(
      f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
  )
  chars_per_token = 4
  # chars_per_token = chars_token_ratio(train_data, tokenizer, args.input_column_name, args.output_column_name)
  # print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

  train_dataset = ConstantLengthDataset(
      tokenizer,
      train_data,
      infinite=True,
      seq_length=args.seq_length,
      chars_per_token=chars_per_token,
  )
  valid_dataset = ConstantLengthDataset(
      tokenizer,
      valid_data,
      infinite=False,
      seq_length=args.seq_length,
      chars_per_token=chars_per_token,
  )
  return train_dataset, valid_dataset


def run_training(args, train_data, val_data, tokenizer_len):
  device_map = "auto"
  world_size = int(os.environ.get("WORLD_SIZE", 1))
  ddp = world_size != 1
  if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size
  print("Loading the model")
  # disable caching mechanism when using gradient checkpointing
  model = AutoModelForCausalLM.from_pretrained(
      args.model_path,
      use_cache=not args.no_gradient_checkpointing,
      load_in_8bit=False,
      torch_dtype=torch.float16,
      device_map=device_map,
  )
  model.resize_token_embeddings(tokenizer_len)
  """
  model = prepare_model_for_int8_training(model)

  lora_config = LoraConfig(r=args.lora_r,
                           lora_alpha=args.lora_alpha,
                           lora_dropout=args.lora_dropout,
                           bias="none",
                           task_type="CAUSAL_LM",
                           target_modules=["c_proj", "c_attn", "q_attn"])

  model = get_peft_model(model, lora_config)
  """

  print_trainable_parameters(model)

  train_data.start_iteration = 0

  print("Starting main loop")
  training_args = TrainingArguments(
      output_dir=args.output_dir,
      dataloader_drop_last=True,
      evaluation_strategy="steps",
      save_strategy='steps',
      #max_steps=args.max_steps,
      num_train_epochs=args.num_train_epochs,
      eval_steps=args.eval_freq,
      save_steps=args.save_freq,
      logging_steps=args.log_freq,
      deepspeed="./deepspeed_config/zero3.json",
      #deepspeed="./deepspeed_config/zero3_no_offload.json",
      #deepspeed="./deepspeed_z3_config_bf16.json",
      per_device_train_batch_size=args.batch_size,
      per_device_eval_batch_size=args.batch_size,
      learning_rate=args.learning_rate,
      #lr_scheduler_type=args.lr_scheduler_type,
      warmup_steps=args.num_warmup_steps,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      gradient_checkpointing=not args.no_gradient_checkpointing,
      bf16=True,
      #fp16=True,
      #bf16=False,
      load_best_model_at_end=True,
      save_total_limit=30,
      weight_decay=args.weight_decay,
      ddp_find_unused_parameters=False,
      report_to="none",
      run_name="starcoder-eng-evol"
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_data,
      eval_dataset=val_data,
     # callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback])
  )
  print("Training...")
  trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
  #trainer.train(resume_from_checkpoint=False)

  print("Saving last checkpoint of the model")
  model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
  print(args.model_path)
  tokenizer = AutoTokenizer.from_pretrained(args.model_path)
  dialogue_tokens = ["<|user|>", "<|assistant|>", "<|end|>"]
  tokenizer.add_special_tokens({"additional_special_tokens": dialogue_tokens})
  tokenizer.save_pretrained(args.output_dir)
  train_dataset, eval_dataset = create_datasets(tokenizer, args)
  run_training(args, train_dataset, eval_dataset, len(tokenizer))


if __name__ == "__main__":
  args = get_args()

  set_seed(args.seed)
  os.makedirs(args.output_dir, exist_ok=True)

  logging.set_verbosity_error()

  main(args)
