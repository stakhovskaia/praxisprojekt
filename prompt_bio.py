import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from datasets import load_dataset
from tqdm import tqdm
from evaluate import load

class BioKeyphraseEvaluator:
    def __init__(self, model_name="microsoft/biogpt"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = next(self.model.parameters()).device

        # Метрики
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")

    def extract_keyphrases(self, sentence, aspect):
        prompt = f"Extract key phrases related to the aspect '{aspect}' from the following sentence: {sentence}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=64)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

    def run_and_evaluate(self, path="test_ds.jsonl", output_path="bionlp_predictions.jsonl"):
        dataset = load_dataset("json", data_files={"test": path})["test"]
        results = []
        references = []
        predictions = []

        for item in tqdm(dataset, desc="Predicting"):
            sentence = item["sentence"]
            aspect = item["aspect"]
            true_kps = item["keyphrases"]["kps"] if isinstance(item["keyphrases"], dict) else item["keyphrases"]
            reference = ", ".join(true_kps)

            prediction = self.extract_keyphrases(sentence, aspect)

            results.append({
                "sentence": sentence,
                "aspect": aspect,
                "prediction": prediction,
                "reference": reference
            })
            predictions.append(prediction)
            references.append(reference)

        # Сохраняем предсказания
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Считаем метрики
        rouge_result = self.rouge.compute(predictions=predictions, references=references)
        bert_result = self.bertscore.compute(predictions=predictions, references=references, lang="en")

        print("\n🎯 Evaluation Results:")
        print(f"ROUGE-1:   {rouge_result['rouge1']:.4f}")
        print(f"ROUGE-2:   {rouge_result['rouge2']:.4f}")
        print(f"ROUGE-Lsum:{rouge_result['rougeLsum']:.4f}")
        print(f"BERTScore (F1): {sum(bert_result['f1']) / len(bert_result['f1']):.4f}")

        return results

if __name__ == "__main__":
    evaluator = BioKeyphraseEvaluator("microsoft/biogpt")
    evaluator.run_and_evaluate("test_ds.jsonl", "bionlp_predictions.jsonl")