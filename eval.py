from transformers import AutoProcessor, Wav2Vec2ForCTC
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from evaluate import load

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True)
wer = load('wer')

def inference(items, outfile, results):

     inputs = processor((items)["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
     outfile.write('file: ' + items['file'] + '\nexpected: ' + items['text'].lower() + '\n')
     results['expected'].append(items['text'].lower())

     with torch.no_grad():
          outputs = model(**inputs)
          logits = outputs.logits

     predicted_ids = torch.argmax(logits, dim=-1)

     transcription = processor.batch_decode(predicted_ids)

     # print(transcription)

     # print(logits)
     # print(logits.size())


     hidden_states = outputs.hidden_states
     # print(hidden_states[0].size())
     # print(hidden_states[0])
     # print('--------')

     # print(model.lm_head(hidden_states[0]))
     # print(model.lm_head(hidden_states[0]).size())

     # print(torch.logit(model.lm_head(hidden_states[0])))
     # print(torch.logit(model.lm_head(hidden_states[0])).size())

     for i in [2,4,6,8,10,12]:
         # print(len(hidden_states[i]))
         predicted_ids = torch.argmax(model.lm_head(hidden_states[i]), dim=-1)
         transcription = processor.batch_decode(predicted_ids)
         outfile.write('\tlayer_' + str(i) + ': ' + transcription[0].lower() + '\n')
         results['layer_' + str(i)].append(transcription[0].lower())
     outfile.write('\n')

werfile = open('wer_results.txt', 'w')

for split in ['validation.clean','validation.other','test.clean','test.other']:
     print(split)
     werfile.write('split: ' + split + '\n')

     dataset = load_dataset("librispeech_asr", split=split, streaming=True)
     sampling_rate = dataset.features["audio"].sampling_rate

     outfile = open(split + '_results.txt', 'w')
     results = {'expected':[],'layer_2':[],'layer_4':[],'layer_6':[],'layer_8':[],'layer_10':[],'layer_12':[]}

     for batch in tqdm(dataset):
          inference(batch, outfile, results)

     for layer in results.keys():
          if layer == 'expected':
               continue
          wer_score = wer.compute(predictions=results[layer], references=results['expected'])
          print(layer + ' WER: ' + str(wer_score))
          werfile.write(layer + ' WER: ' + str(wer_score) + '\n')
     werfile.write('\n')

     outfile.close()

werfile.close()