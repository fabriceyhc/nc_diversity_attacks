import tensorflow as tf
import numpy as np

from keras.models import Model

from sklearn.metrics import accuracy_score

import faulthandler
faulthandler.enable()

# TF
from utils import *

# =============================================================================================== #
# =============================================================================================== #
# =============================================================================================== #
# ========================================= PGD Attacks ========================================= #
# =============================================================================================== #
# =============================================================================================== #
# =============================================================================================== #


# TF 2.x approach
def pgd_attack(model, 
               # modules, 
               # regularizer_weight, 
               inputs, 
               targets, 
               epsilon=None, 
               num_steps=None, 
               step_size=None,
               log_frequency=10):
    
    orig_error = np.sum(np.not_equal(np.argmax(model(inputs), axis=1), targets)) / len(targets)

    depth = np.max(targets) + 1

    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    targets = tf.convert_to_tensor(targets, dtype=tf.int32)
    targets_oh = tf.one_hot(targets, depth)

    random_noise = tf.random.uniform(inputs.shape, -epsilon, epsilon)
    X_pgd = tf.Variable(inputs + random_noise, trainable=True, dtype=tf.float32)

    optimizer = tf.optimizers.SGD()

    for i in range(num_steps):
        with tf.GradientTape() as tape: 
            tape.watch(X_pgd)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(targets_oh, model(X_pgd)))
        gradients = tape.gradient(loss, [X_pgd])
        optimizer.apply_gradients(zip(gradients, [X_pgd]))

        eta = tf.squeeze(step_size * tf.math.sign(gradients))
        X_pgd = tf.Variable(X_pgd + eta, trainable=True, dtype=tf.float32)
        eta = tf.clip_by_value(X_pgd - inputs, -epsilon, epsilon)
        X_pgd = tf.Variable(inputs + eta, trainable=True, dtype=tf.float32)
        X_pgd = tf.Variable(tf.clip_by_value(X_pgd, 0, 1.0), trainable=True, dtype=tf.float32)

        if i % log_frequency == 0:
            print("Loss at step {:03d}: {:.3f}".format(i, loss))

    pgd_error = np.sum(np.not_equal(np.argmax(model(X_pgd.numpy()), axis=1), targets)) / len(targets)

    print('orig_error', orig_error, 'pgd_error', pgd_error)

    return X_pgd, orig_error, pgd_error

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# model = tf.keras.models.load_model(r".\pretrained_models\mnist\test_tf_model.h5")

# n = 1000
# inputs = x_test[:n]
# targets = y_test[:n]

# epsilon=0.25
# num_steps=100
# step_size=0.01
# log_frequency=20

# X_pgd, orig_error, pgd_error = pgd_attack(model, 
#                                           inputs, 
#                                           targets, 
#                                           epsilon, 
#                                           num_steps, 
#                                           step_size,
#                                           log_frequency)


# TF 1.x approach
class LinfPGDAttack_w_Diversity:
    def __init__(self, model, epsilon, k, a, random_start, layer_path, regularizer_weight):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.model_input_channels = self.model.layers[0].input_shape[-1]
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.layer_path = layer_path
        self.regularizer_weight = regularizer_weight
        
        self.x_input = tf.placeholder(tf.float32, shape = [None, 192, 256, 3])
        self.y_input = tf.placeholder(tf.int64, shape = [None])
        self.div_loss = tf.placeholder(tf.float32, shape = ())

        
        x_in = self.x_input[...,:self.model_input_channels]
        y_pred = self.model(x_in)
        
        ### MSE loss ###
        mse_loss = tf.keras.losses.MSE(self.y_input, y_pred)
        
        ### combined loss ###
        loss = mse_loss + self.div_loss

        self.grad = tf.gradients(loss, self.x_input)[0]
        
    def perturb(self, inputs, targets, sess):
        """Given a set of examples (inputs, targets), returns a set of adversarial
           examples within epsilon of inputs in l_infinity norm."""
        if self.rand:
            x = inputs + np.random.uniform(-self.epsilon, self.epsilon, inputs.shape)
            x = np.clip(x, 0, 1) # ensure valid pixel range
        else:
            x = np.copy(inputs)

        for i in range(self.k):
            kl_div = norm_divergence_by_layer(self.model, 
                                              x[...,:self.model_input_channels], 
                                              self.layer_path, 
                                              self.regularizer_weight)
            grad = sess.run(self.grad, feed_dict={self.x_input: inputs,
                                                  self.y_input: targets,
                                                  self.div_loss: kl_div})

            x += self.a * np.sign(grad)

            x = np.clip(x, inputs - self.epsilon, inputs + self.epsilon) 
            x = np.clip(x, 0, 1) # ensure valid pixel range
        return x

from scipy.special import kl_div
def norm_divergence_by_layer(model, data, layer_path, regularizer_weight):

    # layer activation
    input_path = layer_path[0]
    output_path = layer_path[1]
    layer_model = Model(input= model.get_layer(input_path[0]).get_layer(input_path[1]).input,
                        output = model.get_layer(output_path[0]).get_layer(output_path[1]).output)
    layer_activations = np.maximum(layer_model.predict(data), 0) #maximum acts as ReLU

    # normalize over summation (to get a probability density)
    if len(layer_activations.shape) == 1:
        out_norm = (layer_activations / np.sum(layer_activations)) + 1e-20 
    elif len(layer_activations.shape) == 2:
        out_norm = np.sum(layer_activations, axis=0)
        out_norm = (out_norm / np.sum(out_norm)) + 1e-20
    else:
        out_norm = (layer_activations / np.sum(layer_activations)) + 1e-20 

    uniform_tensor = np.ones_like(out_norm)
    uniform_tensor = uniform_tensor / np.sum(uniform_tensor)

    divergence =  np.sum(kl_div(uniform_tensor, out_norm)) * regularizer_weight

    return divergence

# ==================================================================================================== #
# ==================================================================================================== #
# ==================================================================================================== #
# ========================================= Helper Functions ========================================= #
# ==================================================================================================== #
# ==================================================================================================== #
# ==================================================================================================== #

def convert_to_categorical(model_out, num_labels=25):
    min_val = np.min(model_out)
    max_val = np.max(model_out)
    bins = np.linspace(min_val - 1e-5, max_val + 1e-5, num_labels)
    return np.digitize(model_out, bins).reshape(-1)

def eval_performance_reg(model, originals, targets, adversaries, num_labels):
    
    model_input_channels = model.layers[0].input_shape[-1]
    originals = originals[...,:model_input_channels]
    adversaries = adversaries[...,:model_input_channels]
    
    pert_output = model.predict(adversaries)
    orig_output = model.predict(originals)
    
    # MSE
    
    mse = np.square(np.subtract(pert_output, orig_output)).mean()
    
    # Accuracy
    
    classes = convert_to_categorical(targets, num_labels)
    
    pert_pred = convert_to_categorical(pert_output, num_labels)
    orig_pred = convert_to_categorical(orig_output, num_labels)

    pert_acc = accuracy_score(classes, pert_pred)
    orig_acc = accuracy_score(classes, orig_pred)
    
    return mse, pert_acc, orig_acc