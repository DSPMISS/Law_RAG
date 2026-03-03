import os

import evaluate
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model,PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def process_func(examples):
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []
    for idx in range(len(examples["reference"])):
        input_ids_of_single_example, attention_mask_of_single_example, labels_of_single_example = [], [], []
        laws = ""
        for laws in examples["reference"][idx]:
            laws += laws
        instruction = tokenizer("\n".join(["法律条款:" + laws, "客户问题:" + examples["question"][idx]]).strip() + "\n\n法律助手:")
        response = tokenizer(examples["answer"][idx] + tokenizer.eos_token)
        input_ids_of_single_example = instruction["input_ids"] + response["input_ids"]
        attention_mask_of_single_example = instruction["attention_mask"] + response["attention_mask"]
        labels_of_single_example = [-100] * len(instruction["input_ids"]) + response["input_ids"]
        if len(input_ids_of_single_example) > MAX_LENGTH:
            input_ids_of_single_example = input_ids_of_single_example[:MAX_LENGTH]
            attention_mask_of_single_example = attention_mask_of_single_example[:MAX_LENGTH]
            labels_of_single_example = labels_of_single_example[:MAX_LENGTH]
        input_ids.append(input_ids_of_single_example)
        attention_mask.append(attention_mask_of_single_example)
        labels.append(labels_of_single_example)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-350M", cache_dir="./LFM2", low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M", cache_dir="./LFM2")
datasets = load_dataset("json", data_files="../data/真实场景法律咨询/训练数据_带法律依据_92k.json")

# cache_paths = {
#     'train': '../data/cache/mrpc_train_processed.arrow',
#     # 'validation': '../data/cache/mrpc_validation_processed.arrow',
#     # 'test': '../data/cache/mrpc_test_processed.arrow'
# }



tokneized_datasets = datasets.map(
    process_func,
    remove_columns=datasets["train"].column_names,
    keep_in_memory=False,
    batched=True,
    batch_size=500,
)
tokenized_datasets = tokneized_datasets['train'].train_test_split(
    test_size=0.2,      # 测试集比例 20%
    seed=42,           # 随机种子，确保可重复
    shuffle=True       # 是否打乱数据
)

config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "k_proj", "v_proj"],      #仅在这两层设置lora，后一个是正则表达式
                    )
model = get_peft_model(model, config)

# 1. 设置检查点目录
output_dir = ".lora_ckpts"
os.makedirs(output_dir, exist_ok=True)

# 2. 检查是否有已有检查点
last_checkpoint = None
if os.path.exists(output_dir) and os.path.isdir(output_dir):
    # 查找最新的检查点
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
    if checkpoints:
        # 按checkpoint编号排序，取最新的
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        last_checkpoint = os.path.join(output_dir, checkpoints[-1])
        print(f"🔄 发现已有检查点: {last_checkpoint}")
    else:
        print("📭 未发现已有检查点，从头开始训练")
else:
    print("📭 未发现已有检查点，从头开始训练")

train_args = TrainingArguments(
                               num_train_epochs=3,
                               output_dir=output_dir,                #输出文件夹
                               per_device_train_batch_size=1,      #训练时的batchsize
                               per_device_eval_batch_size=1,       #验证时的batchsize
                               logging_steps=100,                    #log打印频率
                               eval_strategy="steps",               #评估频率
                               eval_steps=30000,
                               save_strategy="steps",               #保存频率
                               save_steps=30000,
                               save_total_limit=2,                   #最大保存数
                               learning_rate=1e-5,                  #lr
                               metric_for_best_model="bleu",    #评估指标
                               greater_is_better=True,
                               fp16=True,
                               )

bleu = evaluate.load("bleu")

def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)
    # 解码预测和标签
    # 假设你的tokenizer已经定义好

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # 处理标签：需要移除-100（忽略的token）
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 清理文本：去除多余空格
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # BLEU需要reference是列表的列表格式
    # 每个预测对应一个或多个参考（这里用单个参考）
    references = [[label] for label in decoded_labels]

    # 计算BLEU
    bleu_result = bleu.compute(predictions=decoded_preds, references=references)

    return {
        "bleu": bleu_result["bleu"],
        "precisions": bleu_result["precisions"],
        "brevity_penalty": bleu_result["brevity_penalty"],
        "length_ratio": bleu_result["length_ratio"]
    }

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=eval_metric,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
)



trainer.train(resume_from_checkpoint=last_checkpoint is not None,)

# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = model.num_parameters()  # 或者 sum(p.numel() for p in model.parameters())
#
# print(f"可训练参数量: {trainable_params:,}")
# print(f"总参数量: {total_params:,}")
# print(config)
# print(model)
# for name, param in model.named_parameters():
#     print(name)

