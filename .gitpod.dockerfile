FROM python:3.7

RUN  apt-get update && apt-get install --assume-yes python3-pip nodejs
COPY req.txt .
RUN  pip3 install -r req.txt
WORKDIR "/home"
