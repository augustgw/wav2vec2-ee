from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments
from train_utils import *
from data import *
from tqdm import tqdm
import os 
import torch

ee_alpha = [0.65, 0.70, 0.75, 0.80, 0.85, 1.00] # Dynamic KD application - len must be equal to number of exits 

# torch.set_num_threads(10)

training_args = TrainingArguments(
    output_dir='/workspace/trained_models/dkdee_finetuning',
    evaluation_strategy='no',
    # eval_steps=50,
    # save_total_limit=5,
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=1,
    num_train_epochs=25,
    weight_decay=0.01,
    push_to_hub=False,
    report_to='wandb',
    logging_strategy='epoch',
    dataloader_num_workers=10,
)

# * Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = DKDEEWav2Vec2ForCTC.from_pretrained("trained_models/fixed_ee_finetuning/checkpoint-1406220-epoch-30", ee_alpha=ee_alpha, processor=processor)

# * Train
model.freeze_feature_encoder() # Original Wav2Vec2 paper does not train featurue encoder during fine-tuning
model.train()

trainer = get_trainer(
    model, processor, training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset)

try:
    trainer.train()
except KeyboardInterrupt:
    print('Training early stopped by KeyboardInterrupt.')

# * Save
trainer.save_model()
