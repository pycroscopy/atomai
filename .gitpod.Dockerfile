FROM python:3.7

RUN  apt-get update && apt-get install --assume-yes python3-pip nodejs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN  pip3 install -r requirements.txt
WORKDIR "/home"
