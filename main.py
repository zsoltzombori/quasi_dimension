import tensorflow as tf
import numpy as np
import time
import os
import sys
from scipy.linalg import logm, expm

import networks
import data
import vis

DATASET="mnist"
TRAINSIZE=10000
SEED=None
BN_DO=None  # "BN" (batchnorm), "DO" (dropout), None
BATCH_SIZE=100
DEPTH=4
WIDTH=100
OUTPUT_COUNT = 10
LR=0.001
MEMORY_SHARE=0.25
ITERS=10000
FREQUENCY=1000
AUGMENTATION=False
SESSION_NAME="tmp_{}".format(time.strftime('%Y%m%d-%H%M%S'))
COV_WEIGHT = 0
LATENT_DIM=10
CLASSIFIER=False
HELDOUT_SIZE = 500
AE_TYPE= "conv" # dense, conv
LOG_DIR = "logs/%s" % SESSION_NAME
os.system("rm -rf {}".format(LOG_DIR))

def heuristic_cast(s):
    s = s.strip() # Don't let some stupid whitespace fool you.
    if s=="None":
        return None
    elif s=="True":
        return True
    elif s=="False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

for k, v in [arg.split('=', 1) for arg in sys.argv[1:]]:
    assert v != '', "Malformed command line"
    assert k.startswith('--'), "Malformed arg %s" % k
    k = k[2:]
    assert k in locals(), "Unknown arg %s" % k
    v = heuristic_cast(v)
    print "Changing argument %s from default %s to %s" % (k, locals()[k], v)
    locals()[k] = v


##########################################

(X_train, y_train), (X_devel, y_devel), (X_test, y_test) = data.load_data(DATASET, SEED)

X_train_orig = X_train
y_train_orig = y_train

X_train = X_train[:TRAINSIZE]
y_train = y_train[:TRAINSIZE]

INPUT_SHAPE = X_train.shape[1:]
train_gen = data.classifier_generator((X_train, y_train), BATCH_SIZE, augment=AUGMENTATION)

inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE] + list(INPUT_SHAPE))
if CLASSIFIER:
    output, activations = networks.DenseNet(inputs, DEPTH, WIDTH, BN_DO, OUTPUT_COUNT)
    labels = tf.placeholder(tf.uint8, shape=[BATCH_SIZE])
    expected_output = tf.one_hot(labels, OUTPUT_COUNT)
    main_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=output,
        labels=expected_output
    )
 
else: # autoencoder
    if AE_TYPE == "conv":
        output, activations = networks.ConvAE(inputs, DEPTH, WIDTH, LATENT_DIM)
    elif AE_TYPE == "dense":
        output, activations = networks.DenseAE(inputs, DEPTH, WIDTH, LATENT_DIM)
    else:
        assert False

    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE] + list(INPUT_SHAPE))
    expected_output = labels
    axes = range(1, len(output.shape))
    main_loss = tf.reduce_mean(tf.square(output - expected_output), axis=axes)

main_loss = tf.reduce_mean(main_loss)
loss_list = []
loss_list.append(('main_loss', main_loss))

total_loss = main_loss

cov_ops = []
if COV_WEIGHT > 0: # push the off diagonal elements of the featurewise correlation matrix to zero
    cov_loss = tf.constant(0.0)
    cov_ops = []
    for act in activations:
        feature_count = int(act.shape[1])
        act_centered = act - tf.reduce_mean(act, axis=0)
        for i in range(feature_count):
            for j in range(i, feature_count):
                covariance, cov_op = tf.contrib.metrics.streaming_covariance(act_centered[i], act_centered[j])
                cov_ops.append(cov_op)
                cov_loss += COV_WEIGHT * tf.square(covariance)
    total_loss += cov_loss
    loss_list.append(('cov_loss', cov_loss))


# TODO other losses

loss_list.append(('total_loss', total_loss))

log_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
loss_summaries = []
for (name, loss) in loss_list:
    loss_summaries.append(tf.summary.scalar(name, loss))

merged_loss_summary_op = tf.summary.merge(loss_summaries)


optimizer = tf.train.AdamOptimizer(
    learning_rate=LR
).minimize(total_loss)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = MEMORY_SHARE
session = tf.Session(config=config)
print "NETWORK PARAMETER COUNT", np.sum([np.prod(v.shape) for v in tf.trainable_variables()])

session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

def quasi_dimension(A, calc_eigs=True):
    assert A.ndim == 2
    A = A - np.mean(A, axis=0, keepdims=True)
    if A.shape[0] > A.shape[1]:
        K = np.matmul(A.transpose(), A)
    else:
        K = np.matmul(A, A.transpose())
    T = np.trace(K)
    K_prime= K / T
    if calc_eigs:
        eigvals = np.linalg.eigvalsh(K_prime)
        eigvals = np.clip(eigvals, 1e-10, np.inf)
        eigvals = eigvals
        qd = np.prod(1 / eigvals ** eigvals)
    else:
        log_K_prime = logm(K_prime, disp=False)[0]
        entropy = np.trace(- np.matmul(K_prime, log_K_prime))
        qd = np.exp(entropy)
        qd = np.real(qd)
    qd = np.around(qd, decimals=2)
    return qd

#digit = 5
#heldout_xs = X_train[y_train==digit]
#heldout_xs = heldout_xs[:BATCH_SIZE]
heldout_xs = X_train[:HELDOUT_SIZE]
# def get_qds(X_batch):
#     X_batch_flattened = np.reshape(X_batch, (X_batch.shape[0], -1))
#     qds = [quasi_dimension(X_batch_flattened)]
#     (_activations,) = session.run([activations], feed_dict={inputs:X_batch})
#     for a in _activations:
#         assert a.ndim == 2
#         qds.append(quasi_dimension(a))
#     return qds

def get_qds(Xs):
    global curr_act
    # calculate activations
    act_batches = []
    for i in range(len(Xs) // BATCH_SIZE):
        X_batch = Xs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        (_activations,) = session.run([activations], feed_dict={inputs:X_batch})
        act_batches.append(_activations)

    # put activation batches together
    full_activations=[]
    for i, _ in enumerate(_activations):
        curr_act_list = []
        for act_batch in act_batches:
            curr_act_list.append(act_batch[i])
        curr_act = np.concatenate(curr_act_list, axis=0)
        full_activations.append(curr_act)
        
    # compute qds
    qds = []
    for a in full_activations:
        qds.append(quasi_dimension(a))
    return qds

def evaluate(Xs, ys, BATCH_SIZE):
    nonzeros = []
    for a in activations:
        assert len(a.shape) == 2
        nonzeros.append(np.zeros(int(a.shape[1])))

    eval_gen = data.classifier_generator((Xs, ys), BATCH_SIZE, infinity=False)
    _total_losses = []
    _total_acc = []
    _pred_list = []
    for X_batch, y_batch in eval_gen:
        value_list = session.run([total_loss, output, activations] + list(cov_ops), feed_dict={inputs:X_batch, labels:y_batch if CLASSIFIER else X_batch})
        (_total_loss, predicted, _activations) = value_list[:3]
        _total_acc.append(accuracy(predicted, y_batch))
        _total_losses.append(_total_loss)
        for i, a in enumerate(_activations):
            nonzeros[i] += np.count_nonzero(a, axis=0)

    eval_loss = np.mean(_total_losses)
    eval_acc = np.mean(_total_acc)
    for i, _ in enumerate(nonzeros):
        nonzeros[i] = nonzeros[i] / (len(Xs))
        nonzeros[i] = np.histogram(nonzeros[i], range=(0.0, 1.0))[0]
    return eval_loss, eval_acc, nonzeros


def accuracy(predicted, expected):
    return float(np.sum(np.argmax(predicted, axis=1) == expected)) / len(predicted)

start_time = time.time()

for iteration in xrange(ITERS+1):
    train_data = train_gen.next()

    # training step
    _, _total_loss, predicted, loss_summary = session.run(
        [optimizer, total_loss, output, merged_loss_summary_op],
        feed_dict={inputs: train_data[0], labels: train_data[1] if CLASSIFIER else train_data[0]}
    )
    log_writer.add_summary(loss_summary, iteration)

    # eval step
    if iteration % FREQUENCY == 0:
        train_acc = accuracy(predicted, train_data[1])
        eval_loss, eval_acc, nonzeros = evaluate(X_devel, y_devel, BATCH_SIZE)
        if CLASSIFIER:
            print("{}:\t train acc {},\t dev acc {}").format(iteration, train_acc, eval_acc)
        else:
            print("{}:\t train loss {},\t dev loss {}").format(iteration, _total_loss, eval_loss)
            # X = X_devel[:BATCH_SIZE]
            X = X_train[:BATCH_SIZE]
            (pred,) = session.run([output], feed_dict={inputs:X})
            vis.plotImages(vis.mergeSets((X, pred)), 10, BATCH_SIZE // 20, ("recons", "recons_{}".format(iteration)))

    
    # monitor qds on the training data batch
    if iteration % FREQUENCY == 0:
        qds = get_qds(heldout_xs)
        qds2 = get_qds(X_devel[:HELDOUT_SIZE])
        print qds
        print qds2
                       

print "Total time: {}".format (time.time() - start_time)

