FROM python:3.8 as dashapp
RUN apt-get update && apt-get install ffmpeg -y
COPY . /workdir/
RUN pip install /workdir/
RUN pip install dash
WORKDIR /workdir/app
CMD python dashboard.py

FROM python:3.8 as predapp
RUN apt-get update
RUN pip install --upgrade pip && \
    pip install tensorflow flask
COPY app/ /workdir/app/
COPY model/ /workdir/model/
WORKDIR /workdir/app
CMD python app.py