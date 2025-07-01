import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
import json
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import wandb
from evaluate import load

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeyPhraseExtractor:
    def __init__(self):
        # Загружаем T5-small
        self.model_name = "t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    @staticmethod
    def load_dataset(path="/root/praxis/dataset.jsonl"):
        processed = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    article = entry.get("article", [])
                    summaries = entry.get("summaries", [])
                    for summary in summaries:
                        aspect = summary.get("aspect")
                        sentence_ids = summary.get("sentences", [])
                        kps_list = summary.get("kps", [])
                        if not aspect or not sentence_ids or not kps_list:
                            continue
                        for sent_id, kps in zip(sentence_ids, kps_list):
                            if sent_id < len(article):
                                sentence_text = article[sent_id]
                                processed.append({
                                    "sentence": sentence_text,
                                    "aspect": aspect,
                                    "kps": ", ".join(kps.get("kps", []))  # Объединяем kps в строку
                                })
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
        logger.info(f"Parsed examples: {len(processed)}")
        if not processed:
            raise RuntimeError("No training pairs parsed.")
        
        # Разделяем на train/test (80/20)
        split = int(0.8 * len(processed))
        train_data = processed[:split]
        test_data = processed[split:]
        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "test": Dataset.from_list(test_data)
        })

    def __format_data(self, batch):
        sentences = batch['sentence']
        aspects = batch['aspect']
        kps = batch['kps']
        
        # Формируем входные и целевые тексты
        inputs = [f"extract key phrases for aspect {asp}: {sent}" for asp, sent in zip(aspects, sentences)]
        targets = kps
        
        # Токенизируем
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = self.tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __preprocess_logits_for_metrics(self, logits, labels):
        """
        Обрабатываем логиты для метрик, выбирая токены с максимальной вероятностью.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids

    def __compute_metrics(self, eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        # Декодируем предсказания и эталоны
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Логируем первые 5 примеров в W&B
        table = wandb.Table(columns=["ID", "Prediction", "Reference"])
        for i in range(min(5, len(decoded_preds))):
            table.add_data(i+1, decoded_preds[i], decoded_labels[i])
        wandb.log({"example_predictions": table})
        
        # Вычисляем ROUGE
        rouge = load("rouge")
        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        
        return {
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"]
        }

    def train(self, path="/root/praxis/dataset.jsonl"):
        # Загружаем данные
        dataset = self.load_dataset(path)
        train_dataset = dataset["train"].map(self.__format_data, batched=True)
        test_dataset = dataset["test"].map(self.__format_data, batched=True)
        
        # Выводим первые 3 примера
        for i in range(min(3, len(train_dataset))):
            logger.info(f"Train example {i+1}: {train_dataset[i]['sentence']} -> {train_dataset[i]['kps']}")
        
        # Настраиваем обучение
        training_args = TrainingArguments(
            output_dir="t5_output",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            learning_rate=5e-5,
            fp16=True,
            load_best_model_at_end=True,
            report_to="wandb",
            run_name="t5-keyphrase-extraction"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.__compute_metrics,
            preprocess_logits_for_metrics=self.__preprocess_logits_for_metrics
        )
        
        # Запускаем обучение
        trainer.train()
        trainer.save_model("t5_best_model")

if __name__ == "__main__":
    wandb.init(project="praxis-keyphrase")
    extractor = KeyPhraseExtractor()
    extractor.train()
    