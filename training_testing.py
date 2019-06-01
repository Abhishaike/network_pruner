from keras.models import Model
from keras.layers import Input, Dense, Flatten
import keras
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from pruning_algos import weight_prune, neuron_prune
import tensorflow as tf

#from https://github.com/tensorflow/tensorflow/issues/5354
#No clue why I'm having BLASS errors without this, temporary workaround
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def normalize_images(x_train, x_test):
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    return x_train, x_test


def get_network_architecture():
    '''
    Define internal network architecture. Used in get_trained_model
    '''
    mnist_input = Input(shape=(28,28))
    flattened_mnist_input = Flatten()(mnist_input)
    hidden_layer_1 = Dense(1000, activation='relu',name='Layer_1',kernel_regularizer='l2', use_bias = False)(flattened_mnist_input)
    hidden_layer_2 = Dense(1000, activation='relu',name='Layer_2',kernel_regularizer='l2', use_bias = False)(hidden_layer_1)
    hidden_layer_3 = Dense(500, activation='relu', name='Layer_3',kernel_regularizer='l2', use_bias = False)(hidden_layer_2)
    hidden_layer_4 = Dense(200, activation='relu', name ='Layer_4',kernel_regularizer='l2',use_bias = False)(hidden_layer_3)
    classification = Dense(10, activation='softmax')(hidden_layer_4)
    model_architecture = Model(inputs=mnist_input, outputs=classification)
    return model_architecture

def get_trained_model(x_train, y_train, x_test, y_test):
    '''
    Create baseline fitted model
    '''
    model_architecture = get_network_architecture()
    opt_adam = keras.optimizers.adam(lr=1e-3)
    model_architecture.compile(optimizer=opt_adam, loss='categorical_crossentropy', metrics=['acc'])
    model_architecture.fit(x_train, y_train,
                           batch_size=256,
                           epochs=20,
                           verbose=1,
                           validation_data=(x_test, y_test))
    return model_architecture


def prune_and_save_accuracy(percentile_list, fitted_model):
    '''
    Run model through both types of pruning procedures and save accuracies
    '''
    fitted_weights = fitted_model.get_weights()
    weight_prune_acc = []
    neuron_prune_acc = []
    for percent_pruned in percentile_list:
        weight_pruned_model = weight_prune(fitted_model, percent_prune=percent_pruned)
        weight_prune_acc.append(weight_pruned_model.evaluate(x_test, y_test)[1])
        fitted_model.set_weights(fitted_weights)  # python may be passing pointers to this, setting fitted weights to be safe

        neuron_pruned_model = neuron_prune(fitted_model, percent_prune=percent_pruned)
        neuron_prune_acc.append(neuron_pruned_model.evaluate(x_test, y_test)[1])
        fitted_model.set_weights(fitted_weights)  # python may be passing pointers to this, setting fitted weights to be safe
    return weight_prune_acc, neuron_prune_acc


def plot_accuracy(percentile_list, percentile_name, weight_prune_acc, neuron_prune_acc):
    weight_prune_plot = plt.plot(weight_prune_acc, percentile_list, color='red', label='Weight Pruned Model')
    neuron_prune_plot = plt.plot(neuron_prune_acc, percentile_list, color='blue', label='Neuron Pruned Model')

    plt.xlim(0, 1)  # set labels for the plot to make it easier to read
    plt.ylim(0, 100)
    plt.xlabel('% Pruned')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower left')
    plt.savefig(percentile_name + '.png')
    plt.close()

x_train, y_train, x_test, y_test = load_mnist()
x_train, x_test = normalize_images(x_train, x_test)
fitted_model = get_trained_model(x_train, y_train, x_test, y_test)

all_percentiles = [([0, 25, 50, 60, 70, 80, 90, 95, 97, 99], 'Normal'), #all percentile sets we'd like to go through
                   (list(range(0, 100, 1)), '0_100_1'),
                   (np.linspace(0,100,1000), '0_100_.1')]

for percentile_list, percentile_name in all_percentiles:
    weight_prune_acc, neuron_prune_acc = prune_and_save_accuracy(percentile_list, fitted_model)
    plot_accuracy(percentile_list, percentile_name, weight_prune_acc, neuron_prune_acc)



