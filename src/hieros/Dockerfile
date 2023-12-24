FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y wget unrar ffmpeg libsm6 libxext6

RUN pip install --upgrade pip

COPY ./requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./embodied/scripts/install-atari.sh /workspace/install-atari.sh
RUN bash install-atari.sh
