from transformers import AutoProcessor, Wav2Vec2ForCTC
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from train_utils import *
import sys
import os
import jiwer

if len(sys.argv) != 3:
    sys.exit('Usage: eval.py path/to/checkpoint path/to/results/directory')
else:
    checkpoint_dir = sys.argv[1]
    results_dir = os.getcwd() + '/' + sys.argv[2]
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
model = EEWav2Vec2ForCTC.from_pretrained(checkpoint_dir, output_hidden_states=True)

def inference(items):

    inputs = processor((items)["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    expected = items['text'].lower()

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

        for i in range(6):
            wer_file = open(results_dir + '/entropy_wer/wer_' + str((i+1)*2) + '.txt', 'a+')
        
            # WER
            predicted_ids = torch.argmax(model.decoders[i](hidden_states[(i+1)*2]), dim=-1)
            transcription = processor.batch_decode(predicted_ids)

            wer_out = jiwer.process_words(expected, transcription[0].lower())           
            wer_str = "expect_len: " + str(len(expected.split())) + " " + \
                        "pred_len: " + str(len(transcription[0].split())) + " " + \
                        "hit: " + str(wer_out.hits) + " " + \
                        "sub: " + str(wer_out.substitutions) + " " + \
                        "ins: " + str(wer_out.insertions) + " " + \
                        "del: " + str(wer_out.deletions) + " " + \
                        "wer: " + str(wer_out.wer) + "\n"
            wer_file.write(wer_str)

            wer_file.close()

            entropy_file = open(results_dir + '/entropy_wer/entropy_' + str((i+1)*2) + '.txt', 'a+')

            # Entropy
            logits = model.decoders[i](hidden_states[(i+1)*2]) 
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
            ent_norm = log_probs.size(1) * log_probs.size(2) # num frames * num characters
            
            log_probs = log_probs.squeeze(0)
            probs = torch.exp(log_probs)

            entropy = 0
            for a, b in zip(log_probs, probs):
                entropy += torch.dot(a,b)
            entropy = entropy / ent_norm
            entropy_str = str(entropy.item()) + " " + str(wer_out.wer) + "\n"
            entropy_file.write(entropy_str)

            entropy_file.close()


for split in ['test.clean','test.other']: # ['validation.clean','validation.other','test.clean','test.other']:

    dataset = load_dataset("librispeech_asr", split=split, streaming=True)
    sampling_rate = dataset.features["audio"].sampling_rate

    for batch in tqdm(dataset):
        inference(batch)

    