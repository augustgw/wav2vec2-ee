from data import *
from datasets import load_metric
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback


# Preprocess audio signal
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

# Model config
my_config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base',
                                           gradient_checkpointing=True,
                                           ctc_loss_reduction="mean",
                                           pad_token_id=processor.tokenizer.pad_token_id,
                                           output_hidden_states=True)

# CTC Tokenizer head
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base')

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# Define model
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    config=my_config
)
model.freeze_feature_extractor()
    
# Define TrainingArguments
training_args = TrainingArguments(
    f"training_with_callbacks",
    evaluation_strategy='steps',
    eval_steps=50,  # Evaluation and Save happens every 50 steps
    # Only last 5 models are saved. Older ones are deleted.
    save_total_limit=5,
    learning_rate=2e-5,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    num_train_epochs=10000,
    weight_decay=0.01,
    push_to_hub=False,
    metric_for_best_model='f1',
    load_best_model_at_end=True
)

# Define Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataloader,
    eval_dataset=eval_dataloader,
    tokenizer=processor.feature_extractor,
)

trainer.train()
