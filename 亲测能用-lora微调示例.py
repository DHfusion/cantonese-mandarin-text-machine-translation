'''
tokenizers               0.21.2
scikit-learn             1.7.0
transformers             4.53.1
unsloth                  2025.7.8
unsloth_zoo              2025.7.10
accelerate               1.8.1
torch                    2.7.1
'''


from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
# 加载模型
model_name = r"/data/jiang-1/qwen3-0.6B"
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
# 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
)
# 加载和预处理数据集
#data地址:https://modelscope.cn/datasets/helloworld0/Brain_teasers
dataset = load_dataset("json", data_files=r"/data/jiang-1/data.json", split="train")
train_prompt_style = """下面是一个脑筋急转弯问题，请提供合适的答案，不需要提供思考过程。
### 指令:
你是一个脑筋急转弯专家，请回答以下问题，不需要提供思考过程。
### 问题:
{}
### 回复:
{}"""
def formatting_prompts_func(examples, eos_token):
    inputs = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for inputs, outputs in zip(inputs, outputs):
        text = train_prompt_style.format(inputs, outputs) + eos_token # eos token在training的时候必须要加
        texts.append(text)
    return {
        "text": texts,
    }
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    fn_kwargs={'eos_token': tokenizer.eos_token}, # tokenizer为前面加载model是加载的tokenizer
)
# 配置训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=80,
        learning_rate=5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
# 开始训练
trainer.train()
## 保存LoRA适配器
model.save_pretrained("/data/jiang-1/lora-out")
tokenizer.save_pretrained("/data/jiang-1/lora-out")
## 保存新模型
model.save_pretrained_merged("/data/jiang-1/Qwen3-0.6B-lora", tokenizer, save_method="merged_16bit")
