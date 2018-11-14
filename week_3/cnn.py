#%%
import os
print(os.getcwd())
from Blocks import ReLU, SequentialNN, Dense, Hinge, SGD
from dataset_utils import load_mnist
import numpy as np
from convolution_layer import ConvLayer
from maxpool_layer import MaxPool2x2
from flatten_layer import FlattenLayer

import sys

def iterate_minibatches(x, y, batch_size=16, verbose=True):
    assert len(x) == len(y)

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    for i, start_idx in enumerate(range(0, len(x) - batch_size + 1, batch_size)):
        if verbose:
            print('\rBatch: {}/{}'.format(i + 1, len(x) // batch_size), end='')
            sys.stdout.flush()

        excerpt = indices[start_idx:start_idx + batch_size]
        yield x[excerpt], y[excerpt]

def get_cnn():
    nn = SequentialNN()

    nn.add(ConvLayer(1, 2, filter_size=3)) # The output is of size N_obj 2 28 28
    nn.add(ReLU()) # The output is of size N_obj 2 28 28
    nn.add(MaxPool2x2()) # The output is of size N_obj 2 14 14

    nn.add(ConvLayer(2, 4, filter_size=3)) # The output is of size N_obj 4 14 14
    nn.add(ReLU()) # The output is of size N_obj 4 14 14
    nn.add(MaxPool2x2()) # The output is of size N_obj 4 7 7

    nn.add(FlattenLayer()) # The output is of size N_obj 196
    nn.add(Dense(4 * 7 * 7, 32))
    nn.add(ReLU())
    nn.add(Dense(32, 1))
    return nn

nn = get_cnn()
loss = Hinge()
optimizer = SGD(nn)

train = list(load_mnist(dataset='training', path='.'))
train_images = np.array([im[1] for im in train])
train_targets = np.array([im[0] for im in train])
# We will train a 0 vs. 1 classifier
x_train = train_images[train_targets < 2][:1000]
y_train = train_targets[train_targets < 2][:1000]

y_train = y_train * 2 - 1
y_train = y_train.reshape((-1, 1))

x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((-1, 1, 28, 28))

# It will train for about 5 minutes
num_epochs = 3
batch_size = 32
# We will store the results here
history = {'loss': [], 'accuracy': []}

# `num_epochs` represents the number of iterations
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))

    # We perform iteration a one-by-one iteration of the mini-batches
    for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size):
        # Predict the target value
        y_pred = nn.forward(x_batch)
        # Compute the gradient of the loss
        loss_grad = loss.backward(y_pred, y_batch)
        # Perform backwards pass
        nn.backward(x_batch, loss_grad)
        # Update the params
        optimizer.update_params()

        # Save loss and accuracy values
        history['loss'].append(loss.forward(y_pred, y_batch))
        prediction_is_correct = (y_pred > 0) == (y_batch > 0)
        history['accuracy'].append(np.mean(prediction_is_correct))

    print()

#%%
import matplotlib.pyplot as plt
# Let's plot the results to get a better insight
plt.figure(figsize=(8, 5))

ax_1 = plt.subplot()
ax_1.plot(history['loss'], c='g', lw=2, label='train loss')
ax_1.set_ylabel('loss', fontsize=16)
ax_1.set_xlabel('#batches', fontsize=16)

ax_2 = plt.twinx(ax_1)
ax_2.plot(history['accuracy'], lw=3, label='train accuracy')
ax_2.set_ylabel('accuracy', fontsize=16)
