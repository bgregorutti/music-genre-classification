FROM python38
RUN apt-get install ffmpeg -y
COPY . /workdir/
RUN pip install /workdir/
WORKDIR /workdir/
CMD ["bash"]
