import torch
import math
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
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder, Wav2Vec2EncoderLayer
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
        # Split inputs and labels since they have to be of different lenghts and need different padding methods

        # Preprocessing

        alphabets = ''.join(filter(lambda x: x.isalpha(), list(
            self.processor.tokenizer.decoder.values())))

        input_features = [{"input_values": self.processor(
            audio=feature[0],
            sampling_rate=self.processor.feature_extractor.sampling_rate).input_values[0][0].tolist()}
            for feature in features]
        label_features = [{"input_ids": self.processor(
            text=feature[2].strip().upper() if alphabets.isupper() else feature[2].strip().lower()).input_ids}
            for feature in features]

        # Padding

        batch = self.processor.pad(
            input_features=input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=True,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


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

        # Create new decoders
        self.decoders = nn.ModuleList([self.lm_head for i in range(6)])
        # Initialize with pretrained decoder
        for i in range(len(self.decoders)):
            self.decoders[i].load_state_dict(self.lm_head.state_dict())
        # # Delete pretrained decoder
        # self.lm_head = nn.Identity() # <- When counting parameters, subtract lm_head

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
        for i in range(6):
            # [2,4,6,8,10,12]: layer 0 is not the first layer but the positional embeddings, so skip
            hidden_state = outputs.hidden_states[(i+1)*2]
            hidden_state = self.dropout(hidden_state)

            logits = self.decoders[i](hidden_state)

            if labels is not None:
                if labels.max() >= self.config.vocab_size:
                    raise ValueError(
                        f"Label values must be <= vocab_size: {self.config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(
                        input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(
                    logits, dim=-1, dtype=torch.float32).transpose(0, 1)

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

        with torch.no_grad():
            # self.copy_layer = Wav2Vec2EncoderLayer(config=config)
            # self.copy_layer.load_state_dict(
            #     self.wav2vec2.encoder.layers[11].state_dict())

            # self.copy_decoder = nn.Linear(
            #     in_features=768, out_features=32, bias=True)
            # self.copy_decoder.load_state_dict(
            #     self.lm_head.state_dict())

            self.copy_model = EEWav2Vec2ForCTC.from_pretrained(config.name_or_path)

        # Create new decoders
        self.decoders = nn.ModuleList([self.lm_head for i in range(6)])
        # Initialize with pretrained decoder
        for i in range(len(self.decoders)):
            self.decoders[i].load_state_dict(self.lm_head.state_dict())
        # # Delete pretrained decoder
        # self.lm_head = nn.Identity() # <- When counting parameters, subtract lm_head

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

        with torch.no_grad():
            teacher_outputs = self.copy_model.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            teacher_hidden_states = teacher_outputs.hidden_states

            teacher_ids = torch.argmax(
                self.copy_model.decoders[5](teacher_hidden_states[12]), dim=-1)
            teacher_transcriptions = self.processor.batch_decode(teacher_ids)

            teacher_labels = self.processor(
                text=teacher_transcriptions).input_ids

            # Pad teacher labels (with -100) to max len in batch
            max_label_len = len(max(teacher_labels, key=len))
            for i in range(len(teacher_labels)):
                teacher_labels[i] += [-100] * \
                    (max_label_len - len(teacher_labels[i]))
            teacher_labels = torch.LongTensor(teacher_labels)

            # print(teacher_labels)
            # print(teacher_labels.size())
            # print(labels)
            # print(labels.size())

            # if teacher_labels.max() >= self.config.vocab_size:
            #     raise ValueError(f"Teacher label values must be <= vocab_size: {self.config.vocab_size}")
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(
                    input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(-1)).to(torch.long)
            teacher_labels_mask = teacher_labels >= 0
            teacher_target_lengths = teacher_labels_mask.sum(-1)
            teacher_flattened_targets = teacher_labels.masked_select(
                teacher_labels_mask)
            # print(teacher_flattened_targets.size())

        for i in range(6):
            # [2,4,6,8,10,12]: layer 0 is not the first layer but the positional embeddings, so skip
            hidden_state = outputs.hidden_states[(i+1)*2]
            hidden_state = self.dropout(hidden_state)

            logits = self.decoders[i](hidden_state)

            if labels is not None:
                if labels.max() >= self.config.vocab_size:
                    raise ValueError(
                        f"Label values must be <= vocab_size: {self.config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(
                        input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)
                # print(flattened_targets.size())

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(
                    logits, dim=-1, dtype=torch.float32).transpose(0, 1)

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

                # KD LOSS
                if i < 5:  # Do not compute KD loss for teacher layer
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


class DKDEEWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, ee_alpha, processor):
        super().__init__(config)
        self.ee_alpha = ee_alpha
        self.processor = processor

        # Create new decoders
        self.decoders = nn.ModuleList([self.lm_head for i in range(6)])
        # Initialize with pretrained decoder
        for i in range(len(self.decoders)):
            self.decoders[i].load_state_dict(self.lm_head.state_dict())

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
        kd_loss = list()
        joint_loss = 0

        teacher_ids = torch.argmax(self.lm_head(hidden_states[12]), dim=-1)
        teacher_transcriptions = self.processor.batch_decode(teacher_ids)
        with self.processor.as_target_processor():
            teacher_labels = self.processor(teacher_transcriptions).input_ids

        # Pad teacher labels (with -100) to max len in batch
        max_label_len = len(max(teacher_labels, key=len))
        for i in range(len(teacher_labels)):
            teacher_labels[i] += [-100] * \
                (max_label_len - len(teacher_labels[i]))
        teacher_labels = torch.LongTensor(teacher_labels)

        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(
                input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(
            attention_mask.sum(-1)).to(torch.long)
        teacher_labels_mask = teacher_labels >= 0
        teacher_target_lengths = teacher_labels_mask.sum(-1)
        teacher_flattened_targets = teacher_labels.masked_select(
            teacher_labels_mask)

        for i in range(6):
            # [2,4,6,8,10,12]: layer 0 is not the first layer but the positional embeddings, so skip
            hidden_state = outputs.hidden_states[(i+1)*2]
            hidden_state = self.dropout(hidden_state)

            logits = self.decoders[i](hidden_state)

            if labels is not None:
                if labels.max() >= self.config.vocab_size:
                    raise ValueError(
                        f"Label values must be <= vocab_size: {self.config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(
                        input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(
                    logits, dim=-1, dtype=torch.float32).transpose(0, 1)

                # EE LOSS
                with torch.backends.cudnn.flags(enabled=False):
                    ee_loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )
                    joint_loss += self.ee_alpha[i] * ee_loss

                # KD LOSS
                if i < 5:  # Do not compute KD loss for teacher layer
                    with torch.backends.cudnn.flags(enabled=False):
                        kd_loss = (nn.functional.ctc_loss(
                            log_probs,
                            teacher_flattened_targets,
                            input_lengths,
                            teacher_target_lengths,
                            blank=self.config.pad_token_id,
                            reduction=self.config.ctc_loss_reduction,
                            zero_infinity=self.config.ctc_zero_infinity,
                        ))
                        joint_loss += (1 - self.ee_alpha[i]) * kd_loss

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((joint_loss,) + output) if joint_loss is not None else output

        return CausalLMOutput(
            loss=joint_loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def reset_encoder(self, config):
        self.wav2vec2.encoder = Wav2Vec2Encoder(config)


class ConfidenceWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, processor, inverse_confidence=True):
        super().__init__(config)
        self.processor = processor
        self.inverse_confidence = inverse_confidence

        # Create new decoders
        self.decoders = nn.ModuleList([self.lm_head for i in range(6)])
        # Initialize with pretrained decoder
        for i in range(len(self.decoders)):
            self.decoders[i].load_state_dict(self.lm_head.state_dict())
    
    
    def get_score(self, logits):
        pred_ids = torch.argmax(logits, dim=-1) # shape [1, 329]
        # Take softmax of logits
        scores = torch.nn.functional.log_softmax(logits, dim=-1) # shape [1, 329, 32]
        # Get scores associated with highest probability labels
        pred_scores = scores.gather(-1, pred_ids.unsqueeze(-1))[:, :, 0] # shape [1, 329]
        # Average scores
        score = torch.mean(pred_scores)
        return score
    

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

        # Compute loss for each hidden layer
        loss = 0
        for i in range(6):
            # [2,4,6,8,10,12]: layer 0 is not the first layer but the positional embeddings, so skip
            hidden_state = outputs.hidden_states[(i+1)*2]
            hidden_state = self.dropout(hidden_state)

            logits = self.decoders[i](hidden_state)

            if labels is not None:
                if labels.max() >= self.config.vocab_size:
                    raise ValueError(
                        f"Label values must be <= vocab_size: {self.config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(
                        input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(
                    logits, dim=-1, dtype=torch.float32).transpose(0, 1)
                
                # Compute CTC score
                raw_score = self.get_score(logits)
                conf_score = [max(0.1, math.exp(raw_score))]

                with torch.backends.cudnn.flags(enabled=False):
                    ctc_red_loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction="none",
                        zero_infinity=self.config.ctc_zero_infinity,
                    )
        
                    if self.inverse_confidence:
                        layer_loss = torch.div(ctc_red_loss, torch.Tensor(conf_score).to(self.device))
                    else:
                        layer_loss = torch.mul(ctc_red_loss, torch.Tensor(conf_score).to(self.device))

                    layer_loss = torch.div(layer_loss, target_lengths)
                    layer_loss = torch.mean(layer_loss)

                    loss += layer_loss

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def reset_encoder(self, config):
        self.wav2vec2.encoder = Wav2Vec2Encoder(config)


class ManualExitActivationWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, num_exits=1):
        super().__init__(config)

        self.num_exits = num_exits

        # Create new decoders
        self.decoders = nn.ModuleList([self.lm_head for i in range(self.num_exits)])
        # Initialize with pretrained decoder
        for i in range(len(self.decoders)):
            self.decoders[i].load_state_dict(self.lm_head.state_dict())
        # # Delete pretrained decoder
        # self.lm_head = nn.Identity() # <- When counting parameters, subtract lm_head

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

        # Compute loss for each hidden layer
        loss = 0
        for i in range(self.num_exits):
            # [2,4,6,8,10,12]: layer 0 is not the first layer but the positional embeddings, so skip
            hidden_state = outputs.hidden_states[(i+1)*2]
            hidden_state = self.dropout(hidden_state)

            logits = self.decoders[i](hidden_state)

            if labels is not None:
                if labels.max() >= self.config.vocab_size:
                    raise ValueError(
                        f"Label values must be <= vocab_size: {self.config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(
                        input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(
                    logits, dim=-1, dtype=torch.float32).transpose(0, 1)

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


class ManualExitActivationConfidenceWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, processor, num_exits=1, inverse_confidence=True):
        super().__init__(config)
        self.processor = processor
        self.num_exits = num_exits
        self.inverse_confidence = inverse_confidence

        # Create new decoders
        self.decoders = nn.ModuleList([self.lm_head for i in range(self.num_exits)])
        # Initialize with pretrained decoder
        for i in range(len(self.decoders)):
            self.decoders[i].load_state_dict(self.lm_head.state_dict())

    def get_score(self, logits):
        pred_ids = torch.argmax(logits, dim=-1)  # shape [1, 329]
        # Take softmax of logits
        scores = torch.nn.functional.log_softmax(
            logits, dim=-1)  # shape [1, 329, 32]
        # Get scores associated with highest probability labels
        # shape [1, 329]
        pred_scores = scores.gather(-1, pred_ids.unsqueeze(-1))[:, :, 0]
        # Average scores
        score = torch.mean(pred_scores)
        return score

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

        # Compute loss for each hidden layer
        loss = 0
        
        for i in range(self.num_exits):
            # [2,4,6,8,10,12]: layer 0 is not the first layer but the positional embeddings, so skip
            hidden_state = outputs.hidden_states[(i+1)*2]
            hidden_state = self.dropout(hidden_state)

            logits = self.decoders[i](hidden_state)

            if labels is not None:
                if labels.max() >= self.config.vocab_size:
                    raise ValueError(
                        f"Label values must be <= vocab_size: {self.config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(
                        input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(
                    logits, dim=-1, dtype=torch.float32).transpose(0, 1)

                # Compute CTC score
                raw_score = self.get_score(logits)
                conf_score = [max(0.1, math.exp(raw_score))]

                with torch.backends.cudnn.flags(enabled=False):
                    ctc_red_loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction="none",
                        zero_infinity=self.config.ctc_zero_infinity,
                    )

                    if self.inverse_confidence:
                        layer_loss = torch.div(
                            ctc_red_loss, torch.Tensor(conf_score).to(self.device))
                    else:
                        layer_loss = torch.mul(
                            ctc_red_loss, torch.Tensor(conf_score).to(self.device))

                    layer_loss = torch.div(layer_loss, target_lengths)
                    layer_loss = torch.mean(layer_loss)

                    loss += layer_loss

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
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True)
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
    dataset
) -> List[Dict]:
    inputs = list()
    alphabets = ''.join(filter(lambda x: x.isalpha(), list(
        processor.tokenizer.decoder.values())))
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


def preprocess_item(
    processor: Wav2Vec2Processor,
    item
) -> Dict:
    alphabets = ''.join(filter(lambda x: x.isalpha(), list(
        processor.tokenizer.decoder.values())))
    row = dict()
    array, text = item[0], item[2]
    text = text.upper() if alphabets.isupper() else text.lower()
    row["input_values"] = processor(
        array, sampling_rate=processor.feature_extractor.sampling_rate).input_values[0][0].tolist()
    with processor.as_target_processor():
        row["labels"] = processor(text.strip()).input_ids
    return row


def output_entropy(model):
    # How to compute the entropies from the encoder outputs of an early exit model
    encoder = model.wav2vec2.encoder.layers

    # number_of_frames * number_of_bpetokens
    ent_norm = encoder.size(2) * encoder.size(3)

    i = 0
    for enc in encoder:  # enc are the outputs (softmax) of the encoder layers

        logp = enc.squeeze(0)
        pr = torch.exp(logp)
        entropy = 0
        for a, b in zip(logp, pr):
            entropy += torch.dot(a, b)
        entropy = entropy / ent_norm
        print("entropy of layer[", i, "]:", entropy)
