import tensorflow as tf
import keras

import numpy as np

from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Convolution1D, Convolution2D

# provides a nice UI element when running in a notebook, otherwise use "import tqdm" only
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

def step_through_model(model, prefix=''):
    for i, layer in enumerate(model.layers):
        path = '{}/{}'.format(prefix, layer.name)
        if (isinstance(layer, Dense)
            or isinstance(layer, Convolution1D)
            or isinstance(layer, Convolution2D)):
            yield (path, layer.name, layer)
        elif (isinstance(layer, Sequential)):
            yield from step_through_model(layer, path)

def get_model_layers(model, cross_section_size=0, include_in_out_form=False):
    layer_dict = {}
    i = 0
    
    for (path, name, module) in step_through_model(model):
        layer_dict[str(i) + path] = module
        i += 1
        
    if cross_section_size > 0:
        target_layers = list(layer_dict)[0::cross_section_size] 
        layer_dict = { target_layer: layer_dict[target_layer] for target_layer in target_layers }
        
    if include_in_out_form:
        layer_paths = list(layer_dict.keys())

        first_path_part = None
        second_path_part = None

        input_path = None
        output_path = None

        in_out_layer_paths = []

        for i in range(len(layer_paths)):
            parts = layer_paths[i].split('/')[1:]
            if first_path_part != parts[0]:
                first_path_part = parts[0]
                input_path = parts
                output_path = parts
            elif second_path_part != parts[1]:
                second_path_part = parts[1]
                output_path = parts

            if input_path is not None and output_path is not None:
                in_out_layer_paths.append([input_path, output_path])

        return layer_dict, in_out_layer_paths

    else:

        return layer_dict 

def get_layer_output_sizes(model, layer_name=None):   
    output_sizes = {}
    layer_dict = get_model_layers(model)
 
    for name, layer in layer_dict.items():
        layer_shape = list(layer.output_shape[1:])
        output_sizes[name] = layer_shape
            
    return output_sizes

def get_init_dict(model, init_value=False, layer_name=None): 
    output_sizes = get_layer_output_sizes(model, layer_name)       
    model_layer_dict = {}  
    for layer, output_size in output_sizes.items():
        for index in range(np.prod(output_size)):
            # since we only care about post-activation outputs
            model_layer_dict[(layer, index)] = init_value
    return model_layer_dict

def neurons_covered(model_layer_dict, layer_name=None):
    covered_neurons = len([v for k, v in model_layer_dict.items() if v and (layer_name is None or layer_name in k[0])])
    total_neurons = len([v for k, v in model_layer_dict.items() if layer_name is None or layer_name in k[0]])
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def scale(out, rmax=1, rmin=0):
    output_std = (out - out.min()) / (out.max() - out.min())
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled

def extract_outputs(model, data, in_out_path, force_relu=True):
    model_input_channels = model.layers[0].input_shape[-1]
    data = data[...,:model_input_channels]
    
    input_path = in_out_path[0]
    output_path = in_out_path[1]
    layer_model = Model(input= model.get_layer(input_path[0]).get_layer(input_path[1]).input,
                        output = model.get_layer(output_path[0]).get_layer(output_path[1]).output)
    
    if force_relu:
        layer_out = np.maximum(layer_model.predict(data), 0)
    else:
        layer_out = layer_model.predict(data)
    return layer_out  

def update_coverage(model, data, model_layer_dict, threshold=0., layer_name=None):   
    layer_dict, in_out_layer_paths = get_model_layers(model, include_in_out_form=True) 
    for (layer, module), in_out_path in zip(layer_dict.items(), in_out_layer_paths): 
        layer_outputs = extract_outputs(model, data, in_out_path)
        summed_outputs = np.sum(layer_outputs, axis=0)
        scaled_outputs = scale(summed_outputs)     
        for i, out in enumerate(scaled_outputs.reshape(-1)):
            if out > threshold:
                model_layer_dict[(layer, i)] = True
                
def eval_nc(model, data, threshold=0., layer_name=None):
    model_layer_dict = get_init_dict(model, False)
    update_coverage(model, data, model_layer_dict, threshold=threshold)
    covered_neurons, total_neurons, neuron_coverage = neurons_covered(model_layer_dict, layer_name)
    return covered_neurons, total_neurons, neuron_coverage