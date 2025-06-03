import json
from collections import Counter

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
                                "keyphrases": kps
                            })

            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parse error: {e}")

    return processed


def compute_statistics(pairs):
    total_pairs = len(pairs)
    aspect_counter = Counter([pair["aspect"] for pair in pairs])
    unique_aspects = len(aspect_counter)
    total_sentences = len(set([pair["sentence"] for pair in pairs]))

    print("\nDataset Statistics")
    print("===================")
    print(f"Total pairs: {total_pairs}")
    print("\nExamples:")

    for i, example in enumerate(pairs[:5]):
        print(f"Pair {i + 1}")
        print(f"Sentence: {example['sentence']}")
        print(f"Aspect: {example['aspect']}")

        # Извлекаем только строки ключевых фраз
        kps_extracted = []
        if isinstance(example['keyphrases'], list):
            for kp_item in example['keyphrases']:
                if isinstance(kp_item, dict):
                    kps_extracted.extend(kp_item.get('kps', []))
                else:
                    kps_extracted.append(kp_item)
        else:
            kps_extracted.append(example['keyphrases'])
        print(f"Keyphrases: {kps_extracted}")
        print("-" * 40)

    print(f"Aspects distribution: {dict(aspect_counter)}")
    print(f"Number of unique aspects: {unique_aspects}")
    print(f"Number of unique sentences: {total_sentences}")


if __name__ == "__main__":
    dataset_path = "/root/praxis/dataset.jsonl"
    pairs = load_dataset(dataset_path)
    compute_statistics(pairs)