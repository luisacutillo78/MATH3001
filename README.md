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

import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

* Loading data with Tensorflow
Make sure you have downloaded the data and placed it in data/fashion. Otherwise, Tensorflow will download and use the original MNIST.

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')

data.train.next_batch(BATCH_SIZE)
Note, Tensorflow supports passing in a source url to the read_data_sets. You may use:

data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

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

