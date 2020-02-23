FROM gitpod/workspace-full

RUN  sudo apt-get update && sudo apt-get install --assume-yes python3-pip nodejs
COPY requirements.txt .
RUN  pip3 install -r requirements.txt
WORKDIR "/home"
