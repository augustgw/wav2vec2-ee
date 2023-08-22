from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments
from train_utils import *
from data import *
from tqdm import tqdm
import os 
import torch

ee_alpha = 0.7

torch.set_num_threads(10)

training_args = TrainingArguments(
    output_dir='/workspace/wav2vec2/kdee_finetuning_models',
    evaluation_strategy='no',
    # eval_steps=50,
    # save_total_limit=5,
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=1,
    num_train_epochs=30,
    weight_decay=0.01,
    push_to_hub=False,
    report_to='wandb',
    logging_strategy='epoch',
    dataloader_num_workers=10,
)

# * Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = KDEEWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", ee_alpha=ee_alpha, processor=processor)
# model = KDEEWav2Vec2ForCTC.from_pretrained("ee_loss_finetuning_resume_models/checkpoint-187496", ee_alpha=ee_alpha, processor=processor)

# * Train
model.freeze_feature_encoder() # Original Wav2Vec2 paper does not train featurue encoder during fine-tuning
model.train()

trainer = get_trainer(
    model, processor, training_args,
    train_dataset=preprocess(processor, train_dataset),
    eval_dataset=preprocess(processor, eval_dataset))

try:
    trainer.train()
except KeyboardInterrupt:
    print('Training early stopped by KeyboardInterrupt.')

# * Save
trainer.save_model()
