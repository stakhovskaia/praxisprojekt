import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from evaluate import load
from prompt import prompts
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

class KPSBeforeFineTuning:
    def __init__(self):
        model_name = "unsloth/llama-3-8b-bnb-4bit"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )
        self.model.config.use_cache = True

    def __format_prompt_for_inference(self, batch):
        sentences = batch['sentence']
        aspects = batch['aspect']

        texts = []
        for sentence, aspect in zip(sentences, aspects):
            prompt_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a great assistant for summarizing medical documents. Please follow the instructions strictly.<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompts(sentence, aspect=aspect)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            texts.append(prompt_text)
        return {"text": texts}

    def load_data(self, test_path="test_ds.jsonl"):
        dataset = load_dataset("json", data_files={"test": test_path})["test"]
        dataset = dataset.map(
            self.__format_prompt_for_inference,
            batched=True,
            remove_columns=dataset.column_names,
        )
        return dataset

    def evaluate(self, test_dataset):
        self.model.eval()
        dataloader = DataLoader(test_dataset, batch_size=2)
        predictions = []
        references = []

        raw_data = load_dataset("json", data_files={"test": "test_ds.jsonl"})["test"]

        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating before fine-tuning")):
            inputs = self.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=128)
            decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_preds)

            for j in range(len(decoded_preds)):
                references.append(f"**Key Phrases**: {raw_data[i * 2 + j]['keyphrases']}")

        rouge = load("rouge")
        bert_score = load("bertscore", model_type="distilbert-base-uncased")

        rouge_result = rouge.compute(predictions=predictions, references=references)
        bertscore_result = bert_score.compute(predictions=predictions, references=references, lang="en")

        print("\n======== BEFORE FINE-TUNING METRICS ========")
        print("ROUGE-1:", rouge_result["rouge1"])
        print("ROUGE-2:", rouge_result["rouge2"])
        print("ROUGE-Lsum:", rouge_result["rougeLsum"])
        print("BERTScore F1:", bertscore_result["f1"][0])
        print("===========================================\n")

        wandb.log({
            "before/rouge1": rouge_result["rouge1"],
            "before/rouge2": rouge_result["rouge2"],
            "before/rougeLsum": rouge_result["rougeLsum"],
            "before/bertscore_f1": bertscore_result["f1"][0],
        })

if __name__ == "__main__":
    wandb.init(
        project="kps-keyphrase-extraction",
        name="before-finetuning-eval",
        job_type="evaluation",
    )

    evaluator = KPSBeforeFineTuning()
    test_ds = evaluator.load_data()
    evaluator.evaluate(test_ds)

    wandb.finish()