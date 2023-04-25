FROM pytorch/pytorch

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN apt-get update && apt-get install nano

COPY data.py data.py
COPY train.py train.py
COPY train_utils.py train_utils.py
COPY newtrain.py newtrain.py
COPY conf.py conf.py
COPY vocab.json vocab.json
COPY /util ./util

