from huggingface_hub import notebook_login
from transformers import AutoImageProcessor
from transformers import DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import numpy as np
from datasets import load_dataset



#notebook_login()


# replace this
#plants = load_dataset('plants', data_dir={'train': '/mnt/data6/AI/data/plants/PlantCLEF2024/PlantCLEF2024singleplanttrainingdata/PlantCLEF2024/train', 'test': '/mnt/data6/AI/data/plants/PlantCLEF2024/PlantCLEF2024singleplanttrainingdata/PlantCLEF2024/test', 'validation': '/mnt/data6/AI/data/plants/PlantCLEF2024/PlantCLEF2024singleplanttrainingdata/PlantCLEF2024/validation'})
print("Load dataset")
plants = load_dataset('imagefolder', data_dir='/mnt/data6/AI/data/plants/PlantCLEF2024/PlantCLEF2024singleplanttrainingdata/PlantCLEF2024/')

#plants = plants.train_test_split(test_size=0.2)

print('Label')
labels = plants["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label




checkpoint = "facebook/dinov2-base-imagenet1k-1-layer"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)


print('Normalize')
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


plants = plants.with_transform(transforms)



data_collator = DefaultDataCollator()

import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)



model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)


training_args = TrainingArguments(
    output_dir="plantclef2020",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    tf32=True
)

print('Start Training')
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=plants["train"],
    eval_dataset=plants["test"],
    compute_metrics=compute_metrics,
    tokenizer=image_processor
)

trainer.train()
