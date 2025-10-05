'''
tokenizers               0.21.2
scikit-learn             1.7.0
transformers             4.53.1
unsloth                  2025.7.8
unsloth_zoo              2025.7.10
accelerate               1.8.1
torch                    2.7.1
'''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizer 内部多线程

from unsloth import FastLanguageModel
from trl import SFTTrainer
from trl import SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
max_seq_length=2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/mw/input/qwen38B69506950",
    max_seq_length=2048,  # 支持长上下文
    load_in_4bit=False,    # 4位量化，显存占用降低70%
    load_in_8bit=False,   # 8位量化精度更高，显存需求翻倍
)
print("a")
# 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("b")
# 加载和预处理数据集
#data地址:https://modelscope.cn/datasets/helloworld0/Brain_teasers
dataset = load_dataset("json", data_files=r"/home/mw/project/mt_train+dicnew.json", split="train")

print("c")

train_prompt_style = """{}{}"""
def formatting_prompts_func(examples, eos_token):
    inputs = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for inputs, outputs in zip(inputs, outputs):
        text = train_prompt_style.format(inputs, outputs) + eos_token # eos token在training的时候必须要加
        texts.append(text)
        #print(text)
    return {
        "text": texts,
    }

print("d")    
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    fn_kwargs={'eos_token': tokenizer.eos_token}, # tokenizer为前面加载model是加载的tokenizer
    num_proc=1
)
print("e") 
# 配置训练
 
sft_config = SFTConfig(
    learning_rate=5e-5, # Learning rate for training. 
    num_train_epochs=1, #  Set the number of epochs to train the model.
    per_device_train_batch_size=2, # Batch size for each device (e.g., GPU) during training. 
    gradient_accumulation_steps=8, # Number of steps before performing a backward/update pass to accumulate gradients.
    gradient_checkpointing=False, # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
    logging_steps=2,  # Frequency of logging training progress (log every 2 steps).
    warmup_steps=10,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    output_dir="/home/mw/temp",
    dataset_num_proc = 1
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=sft_config,
)
print("f") 
# 开始训练
trainer.train()
## 保存LoRA适配器
model.save_pretrained("/home/mw/temp/lora_qwen3_newdic")
tokenizer.save_pretrained("/home/mw/temp/lora_qwen3_newdic")
## 保存新模型
model.save_pretrained_merged("/home/mw/temp/lora_qwen3_newdic_merged", tokenizer, save_method="merged_16bit")
