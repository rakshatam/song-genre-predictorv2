import torch
from transformers import ASTForAudioClassification

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
model = ASTForAudioClassification.from_pretrained(model_name, cache_dir="./model_cache")

labels = list(model.config.id2label.values())
with open("classes.txt", "w", encoding="utf-8") as f:
    for label in labels:
        f.write(f"{label}\n")
print(f"Dumped {len(labels)} classes to classes.txt")
