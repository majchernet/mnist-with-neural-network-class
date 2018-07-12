import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from nn import NN


# define model (nodes in each layer)
nn = NN([784,380,160,80,40,20,10])
print ("{}".format(nn.info()))

# Download images and labels
mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)

#train model
for _ in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    X = np.reshape(batch_xs,[-1, 784])
    nn.train( X , batch_ys)

# check accuracy
Xtest = np.reshape(mnist.test.images ,[-1, 784])
print ("Accuracy on test set: %f\n" % nn.test(Xtest, mnist.test.labels))
