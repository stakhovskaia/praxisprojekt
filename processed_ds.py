import json
import os
from sklearn.model_selection import train_test_split

def load_and_process_dataset(input_path):
    processed = []

    with open(input_path, 'r', encoding='utf-8') as f:
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
                                "Kps": kps
                            })

            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parse error: {e}")

    print(f"[DEBUG] Parsed examples: {len(processed)}")
    if not processed:
        raise RuntimeError("No training pairs were parsed. Check dataset structure.")

    return processed

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    input_path = "/root/praxis/dataset.jsonl"
    output_dir = "/root/praxis"

    os.makedirs(output_dir, exist_ok=True)

    data = load_and_process_dataset(input_path)

    # 80/20 split, random_state 
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    save_jsonl(train_data, os.path.join(output_dir, "train_ds.jsonl"))
    save_jsonl(test_data, os.path.join(output_dir, "test_ds.jsonl"))

    print("[INFO] Files saved:")
    print(" - train_ds.jsonl")
    print(" - test_ds.jsonl")

if __name__ == "__main__":
    main()
