from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    Wav2Vec2Config, 
    TrainingArguments
)
from train_utils import *
from data import *
from tqdm import tqdm
import os 
import torch

torch.set_num_threads(10)
torch.manual_seed(999999999)

training_args = TrainingArguments(
    output_dir='/workspace/wav2vec2/ee_loss_training_models',
    evaluation_strategy='no',
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=1,
    num_train_epochs=1000,
    weight_decay=0.01,
    push_to_hub=False,
    report_to='wandb',
    logging_strategy='epoch',
    dataloader_num_workers=10,
)

# * Load model
model = EEWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# * Train
model.freeze_feature_encoder() # Freeze feature encoder (CNN)
model.reset_encoder(config) # Initialize new encoder (Transformer)
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
