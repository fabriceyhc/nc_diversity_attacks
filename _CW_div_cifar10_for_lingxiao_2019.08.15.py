#!/usr/bin/env python
# coding: utf-8

# # CW + Diversity Regularization on CIFAR10

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import pickle
import datetime
import glob
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# custom code imports
# from models import *
from cw_div import *
from neuron_coverage import *
from inception_score import *
from fid_score import *

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

data_dir = "C:\data\CIFAR10"
batch_size_test = 100

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
        # normalize,
        transforms.ToTensor()    
    ])),
    batch_size=batch_size_test, shuffle=False, pin_memory=True)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# # Loading a Pretrained ResNet56
# https://github.com/akamaster/pytorch_resnet_cifar10

from resnet import *

models_dir = 'pretrained_models/cifar10/' 
model = resnet56().cuda()
state_dict = torch.load(models_dir + 'resnet56.th', map_location='cuda')['state_dict'] # best_prec1, state_dict

new_state_dict = {}

for k, v in state_dict.items():
    if 'module' in k:
        k = k.replace('module.', '')
    new_state_dict[k]=v
    
model.load_state_dict(new_state_dict)

# ### Layer Dict
# Required for extracting outputs for diversity regularization. 

layer_dict = get_model_modules(model)
                    
# # Attack Time
def main():

    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    results = []
    save_file_path = "assets/results_cifar10_2019.08.15.pkl"

    # attack params
    search_steps=10
    targeted=False
    norm_type='inf'
    epsilon=10.
    c_range=(1e-3, 1e10)
    max_steps=1000
    abort_early=True
    optimizer_lr=5e-4
    init_rand=False
    log_frequency = 100

    mean = (0.485, 0.456, 0.406) # the mean used in inputs normalization
    std = (0.229, 0.224, 0.225) # the standard deviation used in inputs normalization
    box = (min((0 - m) / s for m, s in zip(mean, std)),
           max((1 - m) / s for m, s in zip(mean, std)))

    n=11
    attack_versions = [cw_div4_attack] # [cw_div1_attack, cw_div2_attack, cw_div3_attack, cw_div4_attack]
    target_layers = list(layer_dict)[0::n]
    reg_weights = [0, 1, 10, 100, 1000, 10000]
    confidences = [0, 5, 10] # [15, 20, 25] # [0, 20, 40]

    # neuron coverage params
    nc_thresholds = 0. # all activations are scaled to (0,1) after relu

    # inception score (is) params
    is_cuda = True
    is_batch_size = 10
    is_resize = True
    is_splits = 10

    # fréchet inception distance score (fid) params
    fid_batch_size = 64
    fid_cuda = True
    real_path = "C:/temp_imgs/cifar/real/"
    fake_path = "C:/temp_imgs/cifar/fake/"

    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    with open('error_log_2019.08.15.txt', 'w') as error_log: 

        for attack in attack_versions:
            for layer_idx in target_layers:
                module = layer_dict[layer_idx]
                for rw in reg_weights:
                    for c in confidences:

                        try:
                        
                            timestamp = str(datetime.datetime.now()).replace(':','.')
                            
                            attack_detail = ['timestamp', timestamp, 
                                             'attack', attack.__name__, 
                                             'layer: ', layer_idx, 
                                             'regularization_weight: ', rw, 
                                             'confidence: ', c]

                            print(*attack_detail, sep=' ')
                            
                            # adversarial attack 
                            adversaries = attack(model, module, rw, inputs, targets, device, targeted, norm_type, epsilon,
                                                 c, c_range, search_steps, max_steps, abort_early, box,
                                                 optimizer_lr, init_rand, log_frequency)
                           
                            # evaluate adversary effectiveness
                            pert_acc, orig_acc = eval_performance(model, inputs, adversaries, targets)
                            sample_3D_images(model, inputs, adversaries, targets, classes)
                            
                            pert_acc = pert_acc.item() / 100.
                            orig_acc = orig_acc.item() / 100.
                            
                            # neuron coverage
                            covered_neurons, total_neurons, neuron_coverage_000 = eval_nc(model, adversaries, 0.00)
                            print('neuron_coverage_000:', neuron_coverage_000)
                            covered_neurons, total_neurons, neuron_coverage_020 = eval_nc(model, adversaries, 0.20)
                            print('neuron_coverage_020:', neuron_coverage_020)
                            covered_neurons, total_neurons, neuron_coverage_050 = eval_nc(model, adversaries, 0.50)
                            print('neuron_coverage_050:', neuron_coverage_050)
                            covered_neurons, total_neurons, neuron_coverage_075 = eval_nc(model, adversaries, 0.75)
                            print('neuron_coverage_075:', neuron_coverage_075)
                            
                            # inception score
                            preprocessed_advs = preprocess_3D_imgs(adversaries)
                            mean_is, std_is = inception_score(preprocessed_advs, is_cuda, is_batch_size, is_resize, is_splits)
                            print('inception_score:', mean_is)
                            
                            # fid score 
                            paths = [real_path, fake_path]
                            
                            # dimensionality = 64
                            target_num = 64
                            generate_imgs(inputs, real_path, target_num)
                            generate_imgs(adversaries, fake_path, target_num)
                            fid_score_64 = calculate_fid_given_paths(paths, fid_batch_size, fid_cuda, dims=64)
                            print('fid_score_64:', fid_score_64)
                            
                            # dimensionality = 2048
                            target_num = 2048
                            generate_imgs(inputs, real_path, target_num)
                            generate_imgs(adversaries, fake_path, target_num)
                            fid_score_2048 = calculate_fid_given_paths(paths, fid_batch_size, fid_cuda, dims=2048)
                            print('fid_score_2048:', fid_score_2048)
                            
                            out = {'timestamp': timestamp, 
                                   'attack': attack.__name__, 
                                   'layer': layer_idx, 
                                   'regularization_weight': rw, 
                                   'confidence': c, 
                                   'adversaries': adversaries,
                                   'pert_acc':pert_acc, 
                                   'orig_acc': orig_acc,
                                   'neuron_coverage_000': neuron_coverage_000,
                                   'neuron_coverage_020': neuron_coverage_020,
                                   'neuron_coverage_050': neuron_coverage_050,
                                   'neuron_coverage_075': neuron_coverage_075,
                                   'inception_score': mean_is,
                                   'fid_score_64': fid_score_64,
                                   'fid_score_2048': fid_score_2048}
                            
                            results.append(out)
                            
                            # save incremental outputs
                            pickle.dump(results, open(save_file_path, "wb"))

                        except Exception as e: 

                            error_log.write("Failed on attack_detail {0}: {1}\n".format(str(attack_detail), str(e)))

                        finally:

                            pass


# df = pd.DataFrame.from_dict(results)
# pd.set_option('display.max_rows', None)
# pd.set_option('precision', 10)
# target_features = ['attack', 'layer', 'regularization_weight', 'confidence', 'orig_acc', 'pert_acc', 'neuron_coverage', 'inception_score', 'fid_score_64', 'fid_score_2048']
# df[target_features]

if __name__ == '__main__':
    main()