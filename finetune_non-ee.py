from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Wav2Vec2Config
from data import *
from train_utils import *


model_config = Wav2Vec2Config(num_hidden_layers=4)

training_args = TrainingArguments(
    output_dir='/workspace/trained_models/non-ee_4layer',
    evaluation_strategy='no',
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1,
    num_train_epochs=40,
    weight_decay=0.01,
    push_to_hub=False,
    report_to='wandb',
    logging_strategy='epoch',
    dataloader_num_workers=36,
    resume_from_checkpoint="trained_models/non-ee_4layer/checkpoint-17578",
    ignore_data_skip=True,
)

# * Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForCTC.from_pretrained("trained_models/non-ee_4layer/checkpoint-17578", config=model_config)

# * Train
model.freeze_feature_encoder() # Original Wav2Vec2 paper does not train feature encoder during fine-tuning
model.train()

trainer = get_trainer(
    model, processor, training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset)

try:
    trainer.train(resume_from_checkpoint="trained_models/non-ee_4layer/checkpoint-17578")
except KeyboardInterrupt:
    print('Training early stopped by KeyboardInterrupt.')

# * Save
trainer.save_model()
