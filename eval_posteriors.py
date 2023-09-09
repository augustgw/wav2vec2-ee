from transformers import AutoProcessor, Wav2Vec2ForCTC
from torchaudio.models.decoder import ctc_decoder
from datasets import load_dataset
from tqdm.auto import tqdm
from evaluate import load
from train_utils import *
import torch
import torchaudio
import jiwer
import sys
import os

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if len(sys.argv) != 4:
     sys.exit('Usage: eval_n_best.py path/to/checkpoint path/to/results/directory n_best')
else:
     checkpoint_dir = sys.argv[1]
     results_dir = os.getcwd() + '/' + sys.argv[2]
     n_best = int(sys.argv[3])
     if not os.path.exists(results_dir):
          os.mkdir(results_dir)

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
model = EEWav2Vec2ForCTC.from_pretrained(checkpoint_dir, output_hidden_states=True)
wer = load('wer')

ctc_decoder = ctc_decoder(
          lexicon = "decoder_utils/lexicon.txt",
          tokens = "decoder_utils/vocab_tokens.txt",
          nbest = n_best,
          log_add = True,
          beam_size = 1500,
          word_score = -0.26,
          blank_token = '<pad>',
          sil_token = '|',
          lm = None,
     )

def inference(items, results, wer_results):

     inputs = processor((items)["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
     expected = items['text'].lower()

     with torch.no_grad():
          outputs = model(**inputs)
          hidden_states = outputs.hidden_states

          for i in range(6):           
               logits = model.decoders[i](hidden_states[(i+1)*2])
               log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
               hyps = ctc_decoder(log_probs)

               # WER
               transcription = ' '.join(hyps[0][0].words)
               wer_out = jiwer.process_words(expected, transcription)
               wer_results[i]['expect_len'].append(str(len(expected.split())))
               wer_results[i]['pred_len'].append(str(len(hyps[0][0].words)))
               wer_results[i]['hits'].append(str(wer_out.hits))
               wer_results[i]['errors'].append(str(wer_out.substitutions + wer_out.insertions + wer_out.deletions))
               wer_results[i]['wer'].append(str(wer_out.wer))

               # posterior
               score = torch.tensor([hyp.score for hyp in hyps[0]])
               softmax = torch.nn.Softmax(dim=0)
               score = softmax(score)
               posterior = torch.max(score)
               results[i].append(str(posterior.item()) + '\n')

results = [[] for x in range(6)]
wer_results = [{'expect_len':[],'pred_len':[],'hits':[],'errors':[],'wer':[]} for x in range(6)]

for split in ['test.clean','test.other']:
     print(split)

     dataset = load_dataset("librispeech_asr", split=split, streaming=True)
     sampling_rate = dataset.features["audio"].sampling_rate

     for batch in tqdm(dataset):
          inference(batch, results, wer_results)

for i in range(6):
     with open(results_dir + '/posterior_' + str((i+1)*2) + '.txt', 'w') as write_file:
          write_file.writelines(results[i])
     with open(results_dir + '/wer_' + str((i+1)*2) + '.txt', 'w') as wer_file:
          layer_wer_results = wer_results[i]          
          for i in range(len(layer_wer_results['expect_len'])):
               wer_str = "expect_len: " + layer_wer_results['expect_len'][i] + " " + \
                         "pred_len: " + layer_wer_results['pred_len'][i] + " " + \
                         "hits: " + layer_wer_results['hits'][i] + " " + \
                         "errors: " + layer_wer_results['errors'][i] + " " + \
                         "wer: " + layer_wer_results['wer'][i] + "\n"
               wer_file.write(wer_str)