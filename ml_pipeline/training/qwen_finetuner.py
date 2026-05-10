import os
import torch
import pandas as pd
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

def finetune_qwen_on_sciq(
    data_path="ml_pipeline/data/sciq_train.json",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    output_dir="saved_models/qwen_ft",
    num_samples=None,
    epochs=1,
    batch_size=1,
    lr=2e-4
):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Run download_dataset.py first.")
        return

    df = pd.read_json(data_path, lines=True)
    if num_samples and num_samples < len(df):
        print(f"Sampling {num_samples} out of {len(df)} rows.")
        df = df.sample(num_samples)
    else:
        print(f"Using full dataset: {len(df)} rows.")
    
    # Format: "Context: {support} \nQuestion: {question}"
    def format_example(row):
        return f"Context: {row['support']}\n\nQuestion: Based on the context, {row['question']}"

    df['text'] = df.apply(format_example, axis=1)
    dataset = Dataset.from_pandas(df[['text']])
    
    print(f"Initializing model and tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # FORCE CPU for demo to avoid CUDA kernel image errors
    device = "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        trust_remote_code=True
    )
    
    # PEFT Config (LoRA)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        use_cpu=True,
        report_to="none",
        eval_strategy="no"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("Starting Qwen fine-tuning...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Qwen on SciQ dataset.")
    parser.add_argument("--data_path", type=str, default="ml_pipeline/data/sciq_train.json")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to use (None for full).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="saved_models/qwen_ft")
    
    args = parser.parse_args()

    # Check for peft before running
    try:
        import peft
        finetune_qwen_on_sciq(
            data_path=args.data_path,
            num_samples=args.num_samples if args.num_samples > 0 else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=args.output_dir
        )
    except ImportError:
        print("PEFT not installed. Please install with 'pip install peft'.")
