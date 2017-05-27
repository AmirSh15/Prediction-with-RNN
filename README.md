# Prediction-with-RNN
Vanila LSTM (unidirectional, bidirectional and multi RNN)
In this project, we use Recurrent neural network to predict our desire time series.
This is made possible by the simple but powerful idea of the [LSTM unit](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf), in which many LSTM units work together to find a model to predict a sequence. If you dont know any detail about LSTM and reccurent neural network, i recommend you to read colah's blog about [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

![lstm3-chain](https://cloud.githubusercontent.com/assets/27130785/26519330/85e14ba2-42d4-11e7-8bf4-d194d23b46dd.png)

We can look at Prediction problem as regression. There are two approaches. First, you can learn your network with one-by-one i-th data as input and (i+1)-th data as label. Second, you can consider a window of data ( (i-N)-th : (i-1)-th ) to predict i-th data. In this project we implement this two approaches with unidirectional LSTM, bidirectional LSTM and multi RNN(LSTM).

For example, I made a simple hand-made time series(sinx) to evaluate our models:

# Prediction-with-unidir

![figure_1](https://cloud.githubusercontent.com/assets/27130785/26519488/019296e6-42d7-11e7-9421-10924cff0d9d.png)
