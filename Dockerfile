FROM bitnami/pytorch

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY data.py data.py
COPY train.py train.py
COPY conf.py conf.py
COPY vocab.json vocab.json
COPY /util ./util
