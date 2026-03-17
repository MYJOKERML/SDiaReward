from datasets import Dataset
import json

def load_dataset_clean(path: str) -> Dataset:
    # Read raw JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Serialize content as strings to avoid Dataset automatic schema alignment
    def serialize(example):
        for key in ["chosen", "rejected"]:
            for turn in example[key]:
                turn["content"] = json.dumps(turn["content"], ensure_ascii=False)
        return example

    processed = [serialize(ex) for ex in data]
    return Dataset.from_list(processed)
