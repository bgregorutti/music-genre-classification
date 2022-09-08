# Music genre classification using Deep Convolutional Network

This repository provides a basic approach for predicting the music genre from WAV files. This is done using a deep convolutional network trained on the well-known [GTZAN dataset](https://www.tensorflow.org/datasets/catalog/gtzan).

A Flask application and a minimal Dash web application run a simple test for prediction, on jazz, reggae and metal musics. The prediction is done in real-time during playing the music.

***References***:
* [Audio Deep Learning Made Simple: Sound Classification, Step-by-Step](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)
* [Tzanetakis, G. and Essl, G. and Cook, P. (2001). Automatic Musical Genre Classification Of Audio Signals. The International Society for Music Information Retrieval.](http://ismir2001.ismir.net/pdf/tzanetakis.pdf)
* [The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use (2013). arXiv preprint](https://arxiv.org/abs/1306.1461)

## How to predict the music genre ?

***Dataset***

The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

The genres are:
* blues
* classical
* country
* disco
* hiphop
* jazz
* metal
* pop
* reggae
* rock

This database is the most widely-used in the benchmark but it is also known that there are many issues (sound quality, repetitions, mislabelling, etc.), see [here](https://arxiv.org/abs/1306.1461). Despite this, this is a good starting point for testing deep learning techniques. See [here](https://github.com/ismir/mir-datasets/blob/master/outputs/mir-datasets.md) for an extensive list of Music Information Retrieval datasets.

***Overall approach***

First, we have to keep in mind that sound can be represented as images, thanks to signal processing techniques such as the well-known **Short Time Fourier Transform**. So the natural way to learning from music is to train a CNN on the spectrogram images derived from the musics.

Our approach if based on a two-blocks convolutional model :
* two 2D convolutional layers (resp. 32 and 128 channels) followed by a max-pooling
* a 20% dropout layer
* a global average pooling layer. This avoids the explosion of the number of parameters in comparison with a simple **flatten** layer.
* a 512 fully connected layer

A deeper network was tested but did not show significantly better results.

![](network.png)

Note that the model is trained on sequences of 3 seconds of music in order to control the size of the images. 


***Model performances***

The model is trained on a sample of 80% of the data randomly chosen. The remaining 20% are used for the validation and test sets.

![](confusion_140622.png)

***Model error during the training***

The following figure represent the accuracy of the model during the training step. A slight overfitting seems to appear after 20 epochs. We should add some regularization layers or we should augment the dataset for better results.

![](history_140622.png)