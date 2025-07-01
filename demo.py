import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import unsloth
import torch
import os
from transformers import EvalPrediction, TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from evaluate import load
from datasets import load_dataset, Dataset, DatasetDict
import json
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KPS:
    def __init__(self):
        # Загружаем модель и токенизатор
        self.model_name = "unsloth/llama-3-8b-bnb-4bit"
        self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=None,
            device_map="auto"
        )

        # Устанавливаем устройство: GPU или CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Применяем шаблон чата
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="llama-3"
        )

    @staticmethod
    def load_custom_dataset(path="/root/praxis/dataset.jsonl"):
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
                                    "Sentences": sentence_text,
                                    "Aspect": aspect,
                                    "Kps": kps.get("kps", [])  # Извлекаем только список kps
                                })

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")

        logger.info(f"Parsed examples: {len(processed)}")
        if not processed:
            raise RuntimeError("No training pairs were parsed. Check dataset structure.")

        # Разделяем на train/test (80/20)
        split = int(0.8 * len(processed))
        train_data = processed[:split]
        test_data = processed[split:]

        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "test": Dataset.from_list(test_data)
        })

    def __formatting_prompts_func(self, batch):
        from prompt import prompts
        sentences = batch['Sentences']
        aspects = batch['Aspect']  # Исправлено: aspcets → aspects
        kps = batch['Kps']

        texts = []
        for sentence, kp, aspect in zip(sentences, kps, aspects):
            message = [
                {"role": "system", "content": "You are a great assistant for summarizing medical documents. Please follow the instructions strictly."},
                {"role": "user", "content": prompts(sentence, aspect=aspect)},
                {"role": "assistant", "content": f"**Key Phrases**: {{'kps': {kp}}}"}
            ]
            texts.append(self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False))

        return {"text": texts}

    def __load_data(self, path="/root/praxis/dataset.jsonl"):
        dataset = KPS.load_custom_dataset(path=path)
        
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
        
        # Выводим первые 3 примера для отладки
        for i in range(min(3, len(train_dataset))):
            logger.info(f"Train example {i+1}: {train_dataset[i]['text']}")
        
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
            loftq_config=None
        )
        logger.info(model.print_trainable_parameters())
        return model, self.tokenizer

    def extract_key_phrases(self, text):
        """
        Извлекает список ключевых фраз из текста модели.
        
        Args:
            text: Строка с выводом модели (например, "**Key Phrases**: {...}")
        
        Returns:
            Список ключевых фраз или пустой список при ошибке
        """
        try:
            text = text.strip()
            logger.debug(f"Input text for extraction: {text}")
            
            # Ищем шаблон "**Key Phrases**: {...}"
            pattern = r"\*\*Key Phrases\*\*:\s*\{.*?\}"
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                logger.warning(f"No Key Phrases pattern found in text: {text}")
                return []
            
            # Извлекаем словарь
            dict_str = match.group(0).split(":", 1)[1].strip()
            
            # Заменяем одинарные кавычки на двойные
            dict_str = dict_str.replace("'", '"')
            
            # Парсим JSON
            kp_dict = json.loads(dict_str)
            kps = kp_dict.get("kps", [])
            
            logger.info(f"Extracted key phrases: {kps}")
            return kps
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}, input: {dict_str}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}, input: {text}")
            return []

    def __compute_metrics(self, eval_pred: EvalPrediction):
        predictions = eval_pred.predictions[0]
        predictions[predictions == -100] = self.tokenizer.pad_token_id

        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # Декодируем предсказания и эталоны
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Извлекаем ключевые фразы
        decoded_predictions_kps_list = [self.extract_key_phrases(pred) for pred in decoded_predictions]
        references_kps_list = [self.extract_key_phrases(ref) for ref in references]

        # Логируем первые 5 примеров
        n = 5
        for i in range(min(n, len(decoded_predictions))):
            logger.info(
                f"Example {i+1}:\n"
                f"Raw Prediction: {decoded_predictions[i]}\n"
                f"Predicted KPs: {decoded_predictions_kps_list[i]}\n"
                f"Reference KPs: {references_kps_list[i]}\n"
            )

        # Проверяем, есть ли предсказания
        if not any(decoded_predictions_kps_list):
            logger.warning("All predicted key phrases are empty!")
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeLsum": 0.0,
                "bertscore": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "recall": 0.0
            }

        # Для ROUGE и BERTScore объединяем ключевые фразы в строки
        decoded_predictions_kps_text = [" ".join(kps) for kps in decoded_predictions_kps_list]
        references_kps_text = [" ".join(kps) for kps in references_kps_list]

        # Загружаем метрики
        rouge = load("rouge")
        bert_score = load("bertscore", model_type="distilbert-base-uncased")

        # Вычисляем ROUGE и BERTScore
        rouge_result = rouge.compute(predictions=decoded_predictions_kps_text, references=references_kps_text)
        bertscore_result = bert_score.compute(predictions=decoded_predictions_kps_text, references=references_kps_text, lang="en")

        # Вычисляем accuracy, f1, recall для ключевых фраз
        all_true_keyphrases = [item for sublist in references_kps_list for item in sublist]
        all_pred_keyphrases = [item for sublist in decoded_predictions_kps_list for item in sublist]

        accuracy = accuracy_score(all_true_keyphrases, all_pred_keyphrases) if all_pred_keyphrases else 0.0
        f1 = f1_score(all_true_keyphrases, all_pred_keyphrases, average="weighted") if all_pred_keyphrases else 0.0
        recall = recall_score(all_true_keyphrases, all_pred_keyphrases, average="weighted") if all_pred_keyphrases else 0.0

        # Возвращаем метрики
        metrics = {
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeLsum": rouge_result["rougeLsum"],
            "bertscore": bertscore_result["f1"][0],
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall
        }
        logger.info(f"Metrics: {metrics}")
        return metrics

    def __preprocess_logits_for_metrics(self, logits, labels):
        """
        Избегаем утечек памяти, выбирая только нужные тензоры.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def train(self, path="/root/praxis/dataset.jsonl"):
        train_dataset, test_dataset = self.__load_data(path=path)
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
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
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

        # Запускаем обучение
        trainer.train(resume_from_checkpoint=False)  # Сбросили обучение
        trainer.save_model("best_models")

if __name__ == "__main__":
    kps = KPS()
    kps.train()