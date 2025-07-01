import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import unsloth
import torch, os
from transformers import (
    EvalPrediction,
    TrainingArguments
)
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from datasets import load_dataset
from evaluate import load
from prompt import prompts

import json

class KPS:
    def __init__(self):
        # load model and tokenizer
        self.model_name = "unsloth/llama-3-8b-bnb-4bit"
        self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=torch.float16,
            device_map={"": 0},   
        )

        # apply chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="llama-3",
        )

    def __formatting_prompts_func(self, batch):
        sentences = batch['sentence']
        aspects = batch['aspect']
        kps = batch['keyphrases']

        texts = []
        for sentences, kps, aspect in zip(sentences, kps, aspects):
            message = [
                {"role": "system", "content": "You are a great assistant for summarizing medical documents. Please follow the instructions strictly."},
                {"role": "user", "content": prompts(sentences, aspect=aspect)},
                {"role": "assistant", "content": f"**Key Phrases**: {kps}"}
            ]
            texts.append(self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False))

        return {"text": texts,}
    
    def __load_data(self, train_path="train_ds.jsonl", test_path="test_ds.jsonl"):
        dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        train_dataset = train_dataset.map(
            self.__formatting_prompts_func,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        test_dataset = test_dataset.map(
            self.__formatting_prompts_func,
            batched=True,
            remove_columns=test_dataset.column_names
        )

        return train_dataset, test_dataset

    def __load_model(self):
        model = unsloth.FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=32,
            use_rslora=True,
            loftq_config=None,
        )
        print(model.print_trainable_parameters())
        return model, self.tokenizer

    def __compute_metrics(self, eval_pred: EvalPrediction):
        predictions = eval_pred.predictions[0]
        predictions[predictions == -100] = self.tokenizer.pad_token_id

        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # Decode predictions to text
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Decode labels to text, ensuring to skip the special token -100 used for ignored indices
        references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        print(f"Number of predictions: {len(decoded_predictions)}, Number of references: {len(references)}")
        
        # load evaluation metric
        rouge = load("rouge")
        bert_score = load("bertscore", model_type="distilbert-base-uncased")
        
        # Calculate metrics
        rouge_result = rouge.compute(predictions=decoded_predictions, references=references)
        bertscore_result = bert_score.compute(predictions=decoded_predictions, references=references, lang="en")
        print("+" * 100)
        print(decoded_predictions[0])
        print("=" * 100)
        print(references[0])

        # Return calculated metrics
        return {
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeLsum": rouge_result["rougeLsum"],
            "bertscore": bertscore_result["f1"][0]
        }

    def __preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def train(self, train_path="train_ds.jsonl", test_path="test_ds.jsonl"):
        train_dataset, test_dataset = self.__load_data(train_path, test_path)
        model, tokenizer = self.__load_model()

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            dataset_text_field="text",
            max_seq_length=4096,
            dataset_num_proc=2,
            packing=False,
            compute_metrics=self.__compute_metrics,
            preprocess_logits_for_metrics=self.__preprocess_logits_for_metrics,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                eval_accumulation_steps=1,
                eval_strategy="steps",
                eval_steps=16,
                gradient_accumulation_steps=1,
                warmup_steps=5,
                num_train_epochs=30,
                learning_rate=1e-5,
                fp16=True,
                bf16=False,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                save_strategy="steps",
                save_steps=16,
                save_total_limit=5,
                load_best_model_at_end=True,
                output_dir="output",
                report_to=None,
            ),
        )

        trainer.train()
        trainer.save_model("best_models")

if __name__ == "__main__":
    kps = KPS()
    kps.train()