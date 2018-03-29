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
    activations.append(output)
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

def ConvAE(inputs, depth, width, latent_dim):
    kernel = (3,3)
    input_shape = inputs.shape
    output = inputs
    activations = []
    for i in range(depth):
        output = tf.layers.conv2d(output, width, kernel, padding="same", name="encoder_{}".format(i))
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        activations.append(tf.contrib.layers.flatten(output))
    pre_latent_shape = output.shape
    pre_latent_count = 1
    for i in range(1, len(pre_latent_shape)):
        pre_latent_count *= int(pre_latent_shape[i])

    output = tf.contrib.layers.flatten(output)
    latent = tf.layers.dense(output, latent_dim, name="latent")
    latent = tf.nn.relu(latent)
    activations.append(latent)

    post_latent = tf.layers.dense(latent, pre_latent_count, name="post_latent")
    post_latent = tf.nn.relu(post_latent)
    output = tf.reshape(post_latent, pre_latent_shape)

    for i in range(depth):
        output = tf.layers.conv2d(output, width, kernel, padding="same", name="decoder_{}".format(i))
        output = tf.nn.relu(output)
        activations.append(tf.contrib.layers.flatten(output))

    output = tf.layers.conv2d(output, input_shape[3], kernel, padding="same", name="output")
    # output = tf.nn.relu(output)
    activations.append(tf.contrib.layers.flatten(output))
    return output, activations

