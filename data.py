import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10
# import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def load_data(dataset, seed=None):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = np.expand_dims(X_train, 3)
        X_test = np.expand_dims(X_test, 3)
    elif dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train = np.expand_dims(X_train, 3)
        X_test = np.expand_dims(X_test, 3)

    elif dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
    else:
        assert False, "Unknown dataset: " + dataset

    # convert brightness values from bytes to floats between 0 and 1:
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # save last 10000 from X_train, y_train for development set
    if dataset == "cifar10" or dataset == "cifar100":
        X_devel = X_test
        y_devel = y_test
    else:
        X_devel = X_train[-10000:]
        y_devel = y_train[-10000:]
        X_train = X_train[:-10000]
        y_train = y_train[:-10000]

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
        sh = np.random.permutation(len(X_train))
        X_train = X_train[sh]
        y_train = y_train[sh]
        sh = np.random.permutation(len(X_test))
        X_test = X_test[sh]
        y_test = y_test[sh]
        np.random.set_state(state)

    return (X_train, y_train), (X_devel, y_devel), (X_test, y_test)

def classifier_generator((xs, ys), batch_size, infinity=True, augment=False):
    if augment:
        datagen = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=False,
            featurewise_std_normalization=True,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.125,
            height_shift_range=0.125,
            horizontal_flip=True,
            vertical_flip=False,
            data_format="channels_last")
        datagen.fit(xs)
    else:
        datagen = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=False,
            featurewise_std_normalization=True,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.,#0.125,
            height_shift_range=0.,#0.125,
            horizontal_flip=False,#True,
            vertical_flip=False,
            data_format="channels_last")
        datagen.fit(xs)

    while True:
        i = 0
        gen = datagen.flow(xs, ys, batch_size=batch_size, shuffle=True)
        while (i+1) * batch_size <= len(xs):
            x, y = gen.next()
            # yield np.reshape(x, [batch_size, -1]), y
            yield x, y
            i += 1
        if not infinity:
            break
