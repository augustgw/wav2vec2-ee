import torch
from torch import nn
import numpy as np
from evaluate import load
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from transformers import (
    Wav2Vec2Processor,
    Trainer,
    Wav2Vec2ForCTC,
    TrainingArguments
    )
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder
from datasets import load_metric, Dataset
from transformers.modeling_outputs import CausalLMOutput
from tqdm import tqdm

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
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

# * Compute metrics
def compute_metrics(pred, processor: Wav2Vec2Processor) -> Dict[str, float]:
    wer_metric = load("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

class EEWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states
        
        # Compute loss for each hidden layer
        loss = 0
        for i in [2,4,6,8,10,12]: # layer 0 is not the first layer but the positional embeddings, so skip
            hidden_state = outputs.hidden_states[i]
            hidden_state = self.dropout(hidden_state)
            
            logits = self.lm_head(hidden_state) 
            
            if labels is not None:
                if labels.max() >= self.config.vocab_size:
                    raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss += nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def reset_encoder(self, config):
        self.wav2vec2.encoder = Wav2Vec2Encoder(config)

class KDEEWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, ee_alpha, processor):
        super().__init__(config)
        self.ee_alpha = ee_alpha
        self.processor = processor

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states
        
        # Compute loss for each hidden layer
        ee_loss = 0
        kd_loss = 0

        # print(self.lm_head(hidden_states[12]).size())
        teacher_ids = torch.argmax(self.lm_head(hidden_states[12]), dim=-1)
        # print(teacher_ids.size())
        teacher_transcriptions = self.processor.batch_decode(teacher_ids)
        # print(teacher_transcriptions))
        with self.processor.as_target_processor():
            teacher_labels = self.processor(teacher_transcriptions).input_ids
        
        # Pad teacher labels (with -100) to max len in batch
        max_label_len = len(max(teacher_labels, key=len))
        for i in range(len(teacher_labels)):
            teacher_labels[i] += [-100] * (max_label_len - len(teacher_labels[i]))
        teacher_labels = torch.LongTensor(teacher_labels)

        # print(teacher_labels)
        # print(teacher_labels.size())
        # print(labels)
        # print(labels.size())
        
        # if teacher_labels.max() >= self.config.vocab_size:
        #     raise ValueError(f"Teacher label values must be <= vocab_size: {self.config.vocab_size}")
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        teacher_labels_mask = teacher_labels >= 0
        teacher_target_lengths = teacher_labels_mask.sum(-1)
        teacher_flattened_targets = teacher_labels.masked_select(teacher_labels_mask)
        # print(teacher_flattened_targets.size())


        for i in [2,4,6,8,10,12]: # layer 0 is not the first layer but the positional embeddings, so skip
            hidden_state = outputs.hidden_states[i]
            hidden_state = self.dropout(hidden_state)
            
            logits = self.lm_head(hidden_state)
            
            if labels is not None:
                if labels.max() >= self.config.vocab_size:
                    raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)
                # print(flattened_targets.size())

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

                # EE LOSS
                with torch.backends.cudnn.flags(enabled=False):
                    ee_loss += nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )

                ## KD LOSS
                if i < 12: # Do not compute KD loss for teacher layer
                    with torch.backends.cudnn.flags(enabled=False):
                        kd_loss += nn.functional.ctc_loss(
                            log_probs,
                            teacher_flattened_targets,
                            input_lengths,
                            teacher_target_lengths,
                            blank=self.config.pad_token_id,
                            reduction=self.config.ctc_loss_reduction,
                            zero_infinity=self.config.ctc_zero_infinity,
                        )

        loss = (self.ee_alpha * ee_loss) + ((1 - self.ee_alpha) * kd_loss)
        
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def reset_encoder(self, config):
        self.wav2vec2.encoder = Wav2Vec2Encoder(config)

def get_trainer(
    model,
    processor: Wav2Vec2Processor,
    training_args: TrainingArguments,
    train_dataset: list,
    eval_dataset: list) -> Trainer:
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=lambda x: compute_metrics(x, processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )
    return trainer

def preprocess(
    processor: Wav2Vec2Processor,
    dataset: Dataset
    ) -> List[Dict]:
    inputs = list()
    alphabets = ''.join(filter(lambda x: x.isalpha(), list(processor.tokenizer.decoder.values())))
    for item in tqdm(dataset):
        row = dict()
        array, text = item[0], item[2]
        text = text.upper() if alphabets.isupper() else text.lower()
        row["input_values"] = processor(
            array, sampling_rate=processor.feature_extractor.sampling_rate).input_values[0][0]
        with processor.as_target_processor():
            row["labels"] = processor(text.strip()).input_ids
        inputs.append(row)
    return inputs
