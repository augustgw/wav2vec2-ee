from transformers import Wav2Vec2Processor, TrainingArguments
from data import *
from train_utils import *

num_exits = 2
inverse_confidence = True

training_args = TrainingArguments(
    output_dir='/workspace/trained_models/manual_activ_inv_conf_2',
    evaluation_strategy='no',
    # eval_steps=50,
    # save_total_limit=5,
    save_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    num_train_epochs=50,
    weight_decay=0.01,
    push_to_hub=False,
    logging_strategy='steps',       
    logging_steps=500,              
    # dataloader_num_workers=10,
    # dataloader_pin_memory=False,
)

# * Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = ManualExitActivationConfidenceWav2Vec2ForCTC.from_pretrained(
    "trained_models/fixed_ee_finetuning/checkpoint-1406220-epoch-30", processor=processor, num_exits=num_exits, inverse_confidence=inverse_confidence)

# * Train
# Original Wav2Vec2 paper does not train feature encoder during fine-tuning
model.freeze_feature_encoder()
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
