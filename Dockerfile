FROM python:3.8 as dashapp
RUN apt-get update && apt-get install ffmpeg -y
COPY . /workdir/
RUN pip install /workdir/
RUN pip install dash
CMD /workdir/app/dashboard.py

FROM python38 as predapp
RUN pip install flask
COPY app/ /workdir/app/
COPY model/ /workdir/model/
CMD /workdir/app/app.py
