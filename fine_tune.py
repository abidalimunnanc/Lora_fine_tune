import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, TaskType

model_name = 'TinyLLama/TinyLlama-1.1B-Chat-v1.0'

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_compute_dtype = torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map = 'auto',
    trust_remote_code = True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

lora_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    target_modules = ['q_proj', 'v_proj'],
    lora_dropout = 0.05,
    bias = 'none',
    task_type = TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)


data = load_dataset('openai/gsm8k', 'main', split='train[:200]')
def tokenize(batch):
    texts = [
        f"### Instruction:\n{inst}\n### Response:\n{out}"
        for inst, out in zip(batch['question'], batch['answer'])
    ]

    tokens = tokenizer(
        texts,
        padding = 'max_length',
        truncation = True,
        max_length = 256,
        return_tensors = 'pt'
    )

    tokens['labels'] = tokens['input_ids'].clone()

    return tokens


tokenized_data = data.map(tokenize, batched=True, remove_columns=data.column_names)

training_args = TrainingArguments(
    output_dir = './tinyllama-lora-tuned',
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate = 1e-3,
    num_train_epochs = 50,
    fp16 = True,
    logging_steps = 20,
    save_strategy = 'epoch',
    report_to = 'none',
    remove_unused_columns = False,      
    label_names = ["labels"]
)


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_data,
    processing_class = tokenizer
)

model.save_pretrained("./tinyllama-lora-tuned-adapter-math")
tokenizer.save_pretrained("./tinyllama-lora-tuned-adapter-math")
