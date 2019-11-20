# MATH3001
Project Title: "Deep Learning for Medical Image Classification".
Teacher: Dr Luisa Cutillo.

This folder will contain some useful material related to this project within the module MATH3001 active at the School of Mathematics, University of Leeds.

## Some useful links:

* YOLO video and tutorial
https://pjreddie.com/darknet/yolo/

* Nando Defreitas Machine Learning Module (Introduction + Lectures >=8):
https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/

* Tutorial:
https://github.com/paras42/Hello_World_Deep_Learning

* Discussion blogs:
https://colah.github.io/
in particular: https://colah.github.io/posts/2014-10-Visualizing-MNIST/

## Let's start playing with data!
We have been looking at the MINST dataset composed of hand written digits with 10 different labels. 
You can download it here: http://yann.lecun.com/exdb/mnist/

We will look at a more recent dataset called Fashion-MNIST dataset.
* you can download this cool dataset here:
https://github.com/zalandoresearch/fashion-mnist
this dataset is called Fashion-MNIST dataset and is a dataset of Zalando's article images. Fashion-MNIST is a 28x28 grayscale image of 70,000 fashion products from 10 categories, with 7,000 images per category. The training set has 60,000 images, and the test set has 10,000 images. 

Fashion-MNIST training and test splits are similar to the original MNIST dataset. It also consists of 10 labels, but instead of handwritten digits, you have 10 different labels of fashion accessories like sandals, shirt, trousers, etc.

Each training and test example is assigned to one of the following labels:

0 T-shirt/top,
1 Trouser,
2 Pullover,
3 Dress,
4 Coat,
5 Sandal,
6 Shirt,
7 Sneaker,
8 Bag,
9 Ankle boot.

Usage
I suggest you clone the Fashion-MINST GitHub repository with the command:
git clone git@github.com:zalandoresearch/fashion-mnist.git

the dataset appears under data/fashion. The https://github.com/zalandoresearch/fashion-mnist repo also contains some scripts for benchmark and visualization!

* Loading data with Python (requires NumPy)
Use utils/mnist_reader in this repo:
```
import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
```
* Loading data with Tensorflow
Make sure you have downloaded the data and placed it in data/fashion. Otherwise, Tensorflow will download and use the original MNIST.
```
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')

data.train.next_batch(BATCH_SIZE)
```
Note, Tensorflow supports passing in a source url to the read_data_sets. You may use:
```
data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
```
* An official Tensorflow tutorial of using tf.keras, a high-level API to train Fashion-MNIST, can be found here:
https://www.tensorflow.org/tutorials/keras/basic_classification

* Try to replicate the datacamp tutorial in https://www.datacamp.com/community/tutorials/introduction-t-sne

## Notes on installing Tensorflow

Install miniconda  https://docs.conda.io/en/latest/miniconda.html 

Launch a conda command line and make sure that the conda command works.  Do the following

```
conda create -n cutillo python=3.6
conda activate cutillo
conda install tensorflow
```

To test this, launch python and do 

```
import tensorflow as tf
```

Tested on windows but should work fine on anything.  

In the above, the environment created is called cutillo (you can use your name).  Every time you start a new session you will need to do 

```
conda activate cutillo 
```

to switch to the version of Python that has Tensorflow installed

# Installing

```
#conda create -n cutillo Python=3.7
#just create this once and istall everything need afyer activating it
conda activate cutillo
conda install tensorflow
conda install jupyter
conda install keras
conda install matplotlib
# once you created this environment, you just need to activate it when you need it and everything you installated will be in it!.```

# Checking it works

Run an example:

```
git clone https://github.com/aymericdamien/TensorFlow-Examples
cd TensorFlow-Examples
cd examples
cd TensorFlow-Examples
python helloworld.py
```

I got 

```
WARNING: Logging before flag parsing goes to stderr.
W0829 13:42:38.948628  9508 deprecation_wrapper.py:119] From helloworld.py:22: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2019-08-29 13:42:38.951885: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  AVX AVX2
To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
2019-08-29 13:42:38.964533: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
b'Hello, TensorFlow!'
```
```
# Back to our dataset!
Run the classification notebook locally or here:
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb
