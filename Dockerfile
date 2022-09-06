FROM python:3.8 as dashapp
COPY . /workdir/
RUN apt-get update && \
    apt-get install ffmpeg -y && \
    pip install /workdir/ dash dash-bootstrap-components && \
    apt autoclean
WORKDIR /workdir/app
CMD python dashboard.py

FROM python:3.8 as predapp
RUN apt-get update && \
    pip install --upgrade pip && \
    pip install tensorflow flask && \
    apt autoclean
COPY app/ /workdir/app/
COPY model/ /workdir/model/
WORKDIR /workdir/app
CMD python app.py