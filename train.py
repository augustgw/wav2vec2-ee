from data import *
from datasets import load_metric
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate
import numpy as np
import torch

# Preprocess audio signal
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

# Model config
my_config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base', 
                                           gradient_checkpointing=True,
                                           ctc_loss_reduction="mean",
                                           pad_token_id=processor.tokenizer.pad_token_id, 
                                           output_hidden_states=True)
# print(my_config)

# # Create model
# model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', config=my_config)

# CTC Tokenizer head
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base')

# Define data collator
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = evaluate.load("wer")

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
model.freeze_feature_encoder()
    
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
    # data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor(return_tensors = 'pt'),
)

trainer.train()