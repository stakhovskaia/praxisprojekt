import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datasets import load_dataset
from tqdm import tqdm
import sys
import wandb
from rouge_score import rouge_scorer

sys.stdout.reconfigure(encoding='utf-8')

class BioKeyphraseExtractor:
    def __init__(self, model_name="microsoft/biogpt"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.model.device

        self.aspect_mapping = {
            "ob": "Objective", "p": "Participants", "i": "Intervention", "c": "Comparator", "o": "Outcomes",
            "f": "Findings", "m": "Medicines", "d": "Treatment Duration", "pe": "Primary Endpoints",
            "s": "Secondary Endpoints", "fo": "Follow-up Duration", "ae": "Adverse Events",
            "r": "Randomization Method", "b": "Blinding Method", "fu": "Funding",
            "rf": "Registration Information", "se": "Secondary Endpoints",
            "fd": "Follow-up Duration", "td": "Treatment Duration"
        }

    def extract_keyphrases(self, sentence, aspect_code):
        aspect = self.aspect_mapping.get(aspect_code, aspect_code)
        prompt = f"Extract key phrases related to {aspect}: {sentence}\nKey phrases:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=4,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "Key phrases:" in decoded:
            return decoded.split("Key phrases:")[-1].strip()
        else:
            return decoded.strip()

    def compute_rouge(self, preds, refs):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}

        for pred, ref in zip(preds, refs):
            result = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)

        return {k: sum(v)/len(v) for k, v in scores.items()}

    def run_on_dataset(self, path="test_ds.jsonl", output_path="bionlp_predictions.jsonl"):
        dataset = load_dataset("json", data_files={"test": path})["test"]
        results = []
        preds = []
        refs = []

        for item in tqdm(dataset, desc="Predicting"):
            sentence = item["sentence"]
            aspect_code = item["aspect"]
            gold = item.get("keyphrases", {}).get("kps", [])

            prediction = self.extract_keyphrases(sentence, aspect_code)
            results.append({
                "sentence": sentence,
                "aspect": aspect_code,
                "prediction": prediction,
                "reference": gold
            })

            preds.append(prediction)
            refs.append(", ".join(gold))
            print(f"[OK] Aspect: {aspect_code}\nPrediction: {prediction}\n---")

        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\nPredictions saved to {output_path}")

        # METRICS
        rouge_results = self.compute_rouge(preds, refs)
        print("\nROUGE scores:")
        for k, v in rouge_results.items():
            print(f"{k}: {v:.4f}")

        # W&B
        wandb.init(project="bio-keyphrase", name="BioGPT-eval", reinit=True)
        wandb.log(rouge_results)

        return results

if __name__ == "__main__":
    extractor = BioKeyphraseExtractor("microsoft/biogpt")
    extractor.run_on_dataset("test_ds.jsonl", "bionlp_predictions.jsonl")