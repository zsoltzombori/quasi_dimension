import tensorflow as tf

def DenseNet(inputs, depth, width, bn_do, output_count):
    activations = []
    zs = []
    output = tf.reshape(inputs, (inputs.shape[0], -1))
    for i in range(depth):
        output = tf.layers.dense(output, width, name="dense_{}".format(i))
        if bn_do == "BN":
            output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        activations.append(output)

    if bn_do == "DO":
        dropout = 0.5 # TODO
        output = tf.nn.dropout(output, dropout)

    output = tf.layers.dense(output, output_count, name="dense_{}".format(depth))
    return output, activations

def DenseAE(inputs, depth, width, latent_dim):
    input_shape = inputs.shape
    input_count = 1
    for i in range(1, len(input_shape)):
        input_count *= int(input_shape[i])
    activations = []
    output = tf.reshape(inputs, (inputs.shape[0], -1))
    for i in range(depth):
        output = tf.layers.dense(output, width, name="encoder_{}".format(i))
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        activations.append(output)

    latent = tf.layers.dense(output, latent_dim, name="latent")
    latent = tf.sigmoid(latent)
    activations.append(latent)

    output = latent
    for i in range(depth):
        output = tf.layers.dense(output, width, name="decoder_{}".format(i))
        # output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        activations.append(output)

    output = tf.layers.dense(output, input_count, name="output")
    output = tf.sigmoid(output)
    activations.append(output)
    output = tf.reshape(output, input_shape)
    return output, activations

# def ConvAE(inputs, depth, width, latent_dim):
#     # todo
#     input_shape = inputs.shape
#     input_count = 1
#     for i in range(1, len(input_shape)):
#         input_count *= int(input_shape[i])
#     activations = []
#     output = tf.reshape(inputs, (inputs.shape[0], -1))
#     for i in range(depth):
#         output = tf.layers.dense(output, width, name="encoder_{}".format(i))
#         output = tf.layers.batch_normalization(output)
#         output = tf.nn.relu(output)
#         activations.append(output)

#     latent = tf.layers.dense(output, latent_dim, name="latent")
#     latent = tf.nn.relu(latent)
#     activations.append(latent)

#     output = latent
#     for i in range(depth):
#         output = tf.layers.dense(output, width, name="decoder_{}".format(i))
#         # output = tf.layers.batch_normalization(output)
#         output = tf.nn.relu(output)
#         activations.append(output)

#     output = tf.layers.dense(output, input_count, name="output")
#     # output = tf.nn.relu(output)
#     activations.append(output)
#     output = tf.reshape(output, input_shape)
#     return output, activations

