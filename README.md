# TASK-2: Extract Key Phrases From Sentence

### 1. Data Preparation

Extract **((sentence, aspect), key-phrases)** pairs from dataset.jsonl. 

```
Elements in kes are words, and neighboring words are a key phrase. The "sentence" is the index of sentence in the article, and the "index" is the index of word in this sentence.

{
    "aspect": "ae",
    "summary": "Since the 5-year analysis, no new safety signals were observed.",
    "kps": [
        {
            "sentence": 9,
            "kps": [
                "no new safety signals"
            ]
        }
    ],
    "sentences": [
        9
    ]
}
```


### 2. Model Input and Output

In this demo, we will fine-tune **"unsloth/Meta-Llama-3.1-8B"**


Model input need to be formatted as: 

"Given some sentences and an aspect, please extract key phrases from the sentence related the given aspect.
Sentences: xxxx. Aspect: xxx
"

Model output need to be key phrases in each given sentence.



### Related Works
- Kazi Saidul Hasan and Vincent Ng. 2014. Automatic Keyphrase Extraction: A Survey of the State of the Art. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1262–1273, Baltimore, Maryland. Association for Computational Linguistics.

- Kamil Bennani-Smires, Claudiu Musat, Andreea Hossmann, Michael Baeriswyl, and Martin Jaggi. 2018. Simple Unsupervised Keyphrase Extraction using Sentence Embeddings. In Proceedings of the 22nd Conference on Computational Natural Language Learning, pages 221–229, Brussels, Belgium. Association for Computational Linguistics.

- https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb