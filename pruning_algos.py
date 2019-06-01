import numpy as np


def weight_prune(fitted_model, percent_prune):
    '''
    Removes weight-level connections depending on the absolute magnitude
    '''
    for layer_num in range(1,5):
        neuron_weights = fitted_model.get_layer(name='Layer_' + str(layer_num)).get_weights()[0] #get single weight layer
        nth_percentiles = np.percentile(np.abs(neuron_weights), percent_prune, axis=0) #get the nth weight percentile for each neuron
        pruned_neuron_weights = neuron_weights * (np.abs(neuron_weights) > nth_percentiles) #bool mask over weights
        fitted_model.get_layer(name='Layer_' + str(layer_num)).set_weights([pruned_neuron_weights]) #replace with pruned weights
    return fitted_model


def neuron_prune(fitted_model, percent_prune):
    '''
    Removes entire neurons depending on the l2 norm of the incoming weights
    '''
    for layer_num in range(1,5):
        neuron_weights = fitted_model.get_layer(name='Layer_' + str(layer_num)).get_weights()[0] #get single weight layer
        l2_norm_weights = np.linalg.norm(neuron_weights, axis = 0) #get the l2 norm of each column (neuron)
        nth_percentile = np.percentile(l2_norm_weights, percent_prune, axis=0) #get the nth weight percentile for all neurons
        pruned_neuron_weights = neuron_weights * (l2_norm_weights > nth_percentile) #bool mask over weights
        fitted_model.get_layer(name='Layer_' + str(layer_num)).set_weights([pruned_neuron_weights]) #replace with pruned weights
    return fitted_model
