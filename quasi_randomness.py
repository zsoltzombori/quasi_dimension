import numpy as np
from scipy.linalg import logm, expm
from keras.datasets import mnist, fashion_mnist

def normalize(A):
    assert np.ndim(A) == 2
    # batch normalization
    # means = np.mean(A, axis=0, keepdims=True)
    # variances = np.var(A, axis=0, keepdims=True)
    # A = (A-means) / np.sqrt(variances + 1e-16)
    mean = np.mean(A)
    variance = np.var(A)
    A = (A-mean) / np.sqrt(variance + 1e-16)
    return A

def quasi_randomness(A):
    assert np.ndim(A) == 2
    A = normalize(A)
    k, n = A.shape
    K = np.matmul(A, A.transpose())
    qr = ((k*n) ** 2) * np.sum(np.square(K)) - (np.sum(A) ** 4)
    return qr / ((k*n) ** 4)

def one_to_zero(size):
    A = np.zeros((size,size))
    for i in range(size):
        for j in range(i):
            A[i][j] = 1.0
    return A

def one_hot(size):
    A = np.zeros((size,size))
    for i in range(size):
        A[i][i] = 1
    return A

# check qr for a set of n unit vectors
if True:
    print "\nCheck qr for a set of n unit vectors"
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000):
        A = one_hot(size)
        qr = quasi_randomness(A)
        print "Size: {}, qr: {}".format(size, qr)

# Size: 10, qr: 90000.0
# Size: 50, qr: 306250000.0
# Size: 100, qr: 9900000000.0
# Size: 200, qr: 3.184e+11
# Size: 500, qr: 3.11875e+13
# Size: 1000, qr: 9.99e+14
# Size: 2000, qr: 3.1984e+16
# Size: 5000, qr: 3.124375e+18

# normalized:
# Size: 10, qr: 0.0009
# Size: 50, qr: 7.84e-06
# Size: 100, qr: 9.9e-07
# Size: 200, qr: 1.24375e-07
# Size: 500, qr: 7.984e-09
# Size: 1000, qr: 9.99e-10
# Size: 2000, qr: 1.249375e-10
# Size: 5000, qr: 7.9984e-12

# check qr for a set of 2*n unit vectors
if True:
    print "\ncheck qr for a set of 2*n unit vectors"
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000):
        A = one_hot(size)
        B = np.concatenate([A, -1 * A], axis=0)
        qr = quasi_randomness(A)
        print "Size: {}, qr: {}".format(size, qr)

# output is same as above

# check qr for gaussian distribution
if True:
    print "\ncheck qr for gaussian distribution"
    dim = 100
    shift = np.random.normal(size=(1, dim))
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
        A = np.random.normal(size=(size, dim))
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        qr = quasi_randomness(A)
        print "Size: {}, qr: {}".format(size, qr)

# Size: 10, qr: 1.09616966152e-05
# Size: 50, qr: 3.00200396206e-06
# Size: 100, qr: 1.96143957023e-06
# Size: 200, qr: 1.50000536489e-06
# Size: 500, qr: 1.19911051055e-06
# Size: 1000, qr: 1.09543645891e-06
# Size: 2000, qr: 1.04898628872e-06
# Size: 5000, qr: 1.02001410143e-06
# Size: 10000, qr: 1.00999040846e-06
# Size: 20000, qr: 1.00487604722e-06

# check qr for an inherently 1 dim dataset for various sizes
if True:
    print "\ncheck qr for an inherently 1 dim dataset for various sizes"
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000):
        A = one_to_zero(size)
        qr = quasi_randomness(A)
        print "Size: {}, qr: {}".format(size, qr)

# Size: 10, qr: 9549375.0
# Size: 50, qr: 4.00328085938e+12
# Size: 100, qr: 1.03329249375e+15
# Size: 200, qr: 2.655973599e+17
# Size: 500, qr: 4.06249351559e+20
# Size: 1000, qr: 1.0408329175e+23
# Size: 2000, qr: 2.6655997336e+25
# Size: 5000, qr: 4.06835930992e+28

# check how qr can be approximated using a subset of the data
if True:
    print "\ncheck how qr can be approximated using a subset of the data"
    size = 200
    A = one_to_zero(size)
    for subset in (10, 30, 50, 100, 200):
        qrs = []
        for i in range(50):
            indices = np.random.permutation(size)
            B = A[indices[:subset]]
            qr = quasi_randomness(B)
            qrs.append(qr)
        qrs= np.array(qrs)
        print "Size: {}, subset: {}, mean: {}, std: {}".format(size, subset, np.mean(qrs), np.std(qrs))

# Size: 200, subset: 10, mean: 1.71020867952e+12, std: 3.69683230702e+11
# Size: 200, subset: 30, mean: 1.39773200925e+14, std: 1.78037397623e+13
# Size: 200, subset: 50, mean: 1.04456805612e+15, std: 1.06814669963e+14
# Size: 200, subset: 100, mean: 1.63247942417e+16, std: 8.42366543962e+14
# Size: 200, subset: 200, mean: 2.655973599e+17, std: 0.0


# check qr on mnist
if True:
    print "\ncheck qr on mnist"
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_train /= 255

    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
        A = X_train[1000:1000+size]
        qr = quasi_randomness(A)
        print "Size: {}, qr: {}".format(size, qr)

# Size: 10, qr: 8.60183141252e+12
# Size: 50, qr: 3.89357645336e+15
# Size: 100, qr: 5.40352526539e+16
# Size: 200, qr: 9.99633902612e+17
# Size: 500, qr: 7.05667835464e+19
# Size: 1000, qr: 9.48912926329e+20
# Size: 2000, qr: 1.44052169798e+22
# Size: 5000, qr: 5.40407526841e+23
# Size: 10000, qr: 8.84678174601e+24
# Size: 20000, qr: 1.37763579139e+26

# check qr on fashion_mnist
if True:
    print "\ncheck qr on fashion_mnist"
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.astype('float32')
    X_train /= 255

    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
        A = X_train[:size]
        qr = quasi_randomness(A)
        print "Size: {}, qr: {}".format(size, qr)

# Size: 10, qr: 6.45220859592e+13
# Size: 50, qr: 3.71875617545e+16
# Size: 100, qr: 5.34262519397e+17
# Size: 200, qr: 7.97280762264e+18
# Size: 500, qr: 3.10623742828e+20
# Size: 1000, qr: 4.96250858316e+21
# Size: 2000, qr: 8.11290787844e+22
# Size: 5000, qr: 3.19661087911e+24
# Size: 10000, qr: 5.14404071965e+25
# Size: 20000, qr: 8.14441163137e+26
