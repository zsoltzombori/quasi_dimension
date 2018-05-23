import numpy as np
from scipy.linalg import logm, expm
from keras.datasets import mnist, fashion_mnist

def quasi_dimension(A):
    K = np.matmul(A, A.transpose())
    K_prime= K / np.trace(K)
    entropy = np.trace(- np.matmul(K_prime, logm(K_prime)))
    qd = np.exp(entropy)
    return qd

def one_to_zero(size):
    A = np.zeros((size,size))
    for i in range(size):
        for j in range(i):
            A[i][j] = 1.0
    return A

def one_hot(size):
    A = np.zeros((size,size))
    for i in range(size):
        A[i][i] = 1.0
    return A

# check qd for a set of n unit vectors
if True:
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000):
        A = one_hot(size)
        qd = quasi_dimension(A)
        print "Size: {}, qd: {}".format(size, qd)

# Size: 10, qd: 10.0
# Size: 50, qd: 50.0
# Size: 100, qd: 100.0
# Size: 200, qd: 200.0
# Size: 500, qd: 500.0
# Size: 1000, qd: 1000.0
# Size: 2000, qd: 2000.0
# Size: 5000, qd: 5000.0

# check qd for an inherently 1 dim dataset for various sizes
if False:
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000):
        A = one_to_zero(size)
        qd = quasi_dimension(A)
        print "Size: {}, qd: {}".format(size, qd)

# Size: 10, qd: 2.15723951064
# Size: 50, qd: 2.3793008326
# Size: 100, qd: 2.40500165818
# Size: 200, qd: 2.4175650281
# Size: 500, qd: 2.42498974572
# Size: 1000, qd: 2.42744205913
# Size: 2000, qd: 2.42866323942
# Size: 5000, qd: 2.42939414209

# check how qd can be approximated using a subset of the data
if False:
    size = 200
    A = one_to_zero(size)
    for subset in (10, 30, 50, 100, 200):
        qds = []
        for i in range(50):
            indices = np.random.permutation(size)
            B = A[indices[:subset]]
            qd = quasi_dimension(B)
            qds.append(qd)
        qds= np.array(qds)
        print "Size: {}, subset: {}, mean: {}, std: {}".format(size, subset, np.mean(qds), np.std(qds))

# Size: 200, subset: 10, mean: 2.03264857439, std: 0.188716511727
# Size: 200, subset: 30, mean: 2.26511261378, std: 0.116539991696
# Size: 200, subset: 50, mean: 2.34617521671, std: 0.0998105281541
# Size: 200, subset: 100, mean: 2.38454085524, std: 0.0674118872019
# Size: 200, subset: 200, mean: 2.4175650281, std: 1.47421565379e-15


# check qd on mnist
if True:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_train /= 255

    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000):
        A = X_train[1000:1000+size]
        qd = quasi_dimension(A)
        print "Size: {}, qd: {}".format(size, qd)

# Size: 10, qd: 5.75850745759
# Size: 50, qd: 11.8502053674
# Size: 100, qd: 14.4751097025
# Size: 200, qd: 17.5796706306
# Size: 500, qd: 20.4906200143
# Size: 1000, qd: (21.1884370967+6.35783104356e-07j)
# Size: 2000, qd: (20.8897976514+1.28197227876e-06j)
# Size: 5000, qd: (22.197434518+1.45175061492e-06j)

# check qd on fashion_mnist
if False:
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.astype('float32')
    X_train /= 255

    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    for size in (10, 50, 100, 200, 500, 1000, 2000, 5000):
        A = X_train[:size]
        qd = quasi_dimension(A)
        print "Size: {}, qd: {}".format(size, qd)

# Size: 10, qd: 3.33004375513
# Size: 50, qd: 4.26551496404
# Size: 100, qd: 4.99672511352
# Size: 200, qd: 5.68861756838
# Size: 500, qd: 6.00010461377
# Size: 1000, qd: (6.17486458389+3.05896709658e-07j)
# Size: 2000, qd: (6.14006476095+6.80514402012e-07j)
# Size: 5000, qd: (6.25355646503+1.47959989533e-06j)
