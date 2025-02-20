import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
import jsonlines
from pathlib import Path
import logging
import wandb
from dataclasses import dataclass
from typing import Dict, List

logging.basicConfig(level=logging.INFO)


@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 500


class CodeDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, max_length: int):
        self.examples = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                self.examples.append(self.format_example(obj))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_example(self, example: Dict) -> str:
        return f"Context: {example['explanation']}\nCode: {example['code']}"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        text = self.examples[i]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }


def train_model(config: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    data_dir = Path('data/processed')
    train_dataset = CodeDataset(data_dir / 'train.jsonl', tokenizer, config.max_length)
    val_dataset = CodeDataset(data_dir / 'val.jsonl', tokenizer, config.max_length)

    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="wandb",
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    # Save final model
    model.save_pretrained("models/final")
    tokenizer.save_pretrained("models/final")


if __name__ == "__main__":
    wandb.init(project="code-completion-ft")
    config = TrainingConfig()
    train_model(config)