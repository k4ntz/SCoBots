# this dockerfile once scobi is pip installable from the web (public github, pypi etc)
FROM ubuntu:20.04

WORKDIR /workdir

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-venv python3-tk git
RUN python3 -m venv .venv
ENV PATH="/workdir/.venv/bin:$PATH"


COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install opencv-python-headless

COPY train.py train.py
RUN mkdir "baselines_logs"
RUN mkdir "baselines_checkpoints"
RUN mkdir "baselines_focusfiles"
ENTRYPOINT ["python3", "-u" , "train.py"]