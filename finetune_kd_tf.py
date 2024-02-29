import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments

from data import *
from train_utils import *


ee_alpha = 0.3

# torch.set_num_threads(10)

training_args = TrainingArguments(
    output_dir='/workspace/trained_models/newkd_ee0.3_finetuning',
    evaluation_strategy='no',
    # eval_steps=50,
    # save_total_limit=5,
    save_strategy = 'epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    num_train_epochs=50,
    weight_decay=0.01,
    push_to_hub=False,
    # report_to='wandb',
    logging_strategy='epoch',
    # dataloader_num_workers=10,
    # dataloader_pin_memory=False,
)

# * Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = NewKDEEWav2Vec2ForCTC.from_pretrained(
    "trained_models/fixed_ee_finetuning/checkpoint-1406220-epoch-30", ee_alpha=ee_alpha, processor=processor)

# * Train
model.freeze_feature_encoder() # Original Wav2Vec2 paper does not train featurue encoder during fine-tuning
model.train()

trainer = get_trainer(
    model, processor, training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset)

try:
    trainer.train(resume_from_checkpoint="trained_models/fixed_kdee_finetuning/checkpoint-421866-epoch-18")
except KeyboardInterrupt:
    print('Training early stopped by KeyboardInterrupt.')

# * Save
trainer.save_model()
