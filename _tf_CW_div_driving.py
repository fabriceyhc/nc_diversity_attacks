# PGD + Diversity Regularization on MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, ImageFolder

import numpy as np
import matplotlib.pyplot as plt

import traceback
import warnings
warnings.filterwarnings('ignore')

import datetime
import glob
import os
import pickle

from models import *
from utils import *
from div_attacks_tf import *
from neuron_coverage_tf import *
from inception_score import *
from fid_score import *

# TF

import tensorflow as tf
import keras

print('tf', tf.version.VERSION)
print('keras', keras.__version__)

# check if CUDA is available
device = torch.device("cpu")
use_cuda = False
if torch.cuda.is_available():
    print('CUDA is available!')
    device = torch.device("cuda")
    use_cuda = True
else:
    print('CUDA is not available.')

random_seed = 1
torch.manual_seed(random_seed)

# TF1 Specific Utils

data_dir = r'C:\data\udacity_self_driving_car'
targets_file = 'targets.csv'
batch_size = 32

dataset = car_loader(target_csv_file=os.path.join(data_dir, targets_file),
                    img_dir=os.path.join(data_dir, 'data'),
                    device=device,
                    num_classes=25,
                    transform=transforms.Compose([transforms.ToTensor(),
                                                  transforms.ToPILImage(),
                                                  transforms.Resize((192, 256)),
                                                  transforms.ToTensor()]))

test_loader = DataLoader(dataset, batch_size=batch_size)

# Generate a custom batch to ensure that each "class" of steering angles is equally represented

num_per_class = 4
class_distribution = torch.ones(dataset.num_classes) * num_per_class
inputs, targets, classes = generate_batch_reg(dataset, class_distribution, device)

np_inputs = inputs.clone().permute(0,2,3,1).cpu().detach().numpy()
np_targets = targets.clone().cpu().detach().numpy()
np_classes = classes.clone().cpu().detach().numpy()

# # Load Pretrained Models if available

rambo = keras.models.load_model(r".\pretrained_models\driving\rambo_model.hdf5")

# # Attack Time
def main():

    models = [rambo]

    # attack params
    epsilon = 100.
    num_steps = 10
    step_size = 0.01
    log_frequency = 100

    # primary evaluation criteria
    attack_versions = [LinfPGDAttack_w_Diversity]
    reg_weights = [0, 1, 10, 100, 1000, 10000, 100000, 1000000]
    epsilons = [0.1, 0.2, 0.3]

    # neuron coverage params
    nc_threshold = 0. # all activations are scaled to (0,1) after relu

    # inception score (is) params
    is_cuda = use_cuda
    is_batch_size = 10
    is_resize = True
    is_splits = 10

    # fréchet inception distance score (fid) params
    real_path = "C:/temp_imgs/mnist/real_pgd_driving/"
    fake_path = "C:/temp_imgs/mnist/fake_pgd_driving/"
    fid_batch_size = 64
    fid_cuda = use_cuda                    

    with open('logs/pgd_mnist_error_log_2020.02.07.txt', 'w') as error_log: 

        for model in models:

            results = []
            model_name = model.__class__.__name__
            save_file_path = "assets/tf_pgd_results_driving_" + model_name + "_2020.02.07.pkl"   
            timestamp = str(datetime.datetime.now()).replace(':','.')

            # neuron coverage
            covered_neurons, total_neurons, neuron_coverage_000 = eval_nc(model, np_inputs, 0.00)
            print('neuron_coverage_000:', neuron_coverage_000)
            covered_neurons, total_neurons, neuron_coverage_020 = eval_nc(model, np_inputs, 0.20)
            print('neuron_coverage_020:', neuron_coverage_020)
            covered_neurons, total_neurons, neuron_coverage_050 = eval_nc(model, np_inputs, 0.50)
            print('neuron_coverage_050:', neuron_coverage_050)
            covered_neurons, total_neurons, neuron_coverage_075 = eval_nc(model, np_inputs, 0.75)
            print('neuron_coverage_075:', neuron_coverage_075)
            
            # inception score
            preprocessed_inputs = preprocess_3D_imgs(inputs)
            mean_is, std_is = inception_score(preprocessed_inputs, is_cuda, is_batch_size, is_resize, is_splits)
            print('inception_score:', mean_is)

            init = {'desc': 'Initial inputs and targets.', 
                    'timestamp': timestamp, 
                    'attack': 'NA', 
                    'model': model_name, 
                    'layer': 'NA', 
                    'epsilon': 'NA',
                    'regularization_weight': 'NA', 
                    'inputs': np_inputs,
                    'targets': np_targets,
                    'binned_targets': np_classes,
                    'adversaries': 'NA',
                    'mse': 'NA',
                    'pert_acc': 'NA', 
                    'orig_acc': 'NA',
                    'attack_success_rate': 'NA',
                    'neuron_coverage_000': neuron_coverage_000,
                    'neuron_coverage_020': neuron_coverage_020,
                    'neuron_coverage_050': neuron_coverage_050,
                    'neuron_coverage_075': neuron_coverage_075,
                    'inception_score': mean_is,
                    'fid_score_64': 'NA',
                    'fid_score_2048': 'NA',
                    'output_diversity': 'NA',
                    'output_diversity_pct': 'NA'}

            results.append(init)

            # get layer paths for activation extraction
            layer_dict, in_out_layer_paths = get_model_layers(model, cross_section_size=0, include_in_out_form=True)

            n=3
            in_out_layer_paths = in_out_layer_paths[::n]

            for attack in attack_versions:
                for in_out_path in in_out_layer_paths:
                    for rw in reg_weights:
                        for e in epsilons:

                            try:
                            
                                timestamp = str(datetime.datetime.now()).replace(':','.')
                                
                                attack_detail = ['model', model_name,
                                                 'timestamp', timestamp, 
                                                 'attack', attack.__name__, 
                                                 'layer: ', in_out_path, 
                                                 'regularization_weight: ', rw, 
                                                 'epsilon: ', e]

                                print(*attack_detail, sep=' ')
     
                                # adversarial attack 
                                
                                pgd_attack = attack(model = model,
                                                    epsilon = e,
                                                    k = num_steps,
                                                    a = step_size,
                                                    random_start = True,
                                                    layer_path = in_out_path,
                                                    regularizer_weight = rw)

                                print('generating adversaries...')
                                with tf.Session() as sess:
                                    sess.run(tf.global_variables_initializer())
                                    np_adversaries = pgd_attack.perturb(np_inputs, np_targets, sess)
                               
                                # evaluate adversary effectiveness
                                mse, pert_acc, orig_acc = eval_performance_reg(model, np_inputs, np_targets, np_adversaries, dataset.num_classes)
                                # sample_3D_images_reg(model, inputs, adversaries, targets, classes, dataset)

                                print('MSE: {:.4f}'.format(mse))
                                print('Perturbed Accuracy: {:.2f}%'.format(pert_acc * 100))
                                print('Original Accuracy: {:.2f}%'.format(orig_acc * 100))
                                
                                attack_success_rate = 1 - pert_acc

                                # neuron coverage
                                covered_neurons, total_neurons, neuron_coverage_000 = eval_nc(model, np_adversaries, 0.00)
                                print('neuron_coverage_000:', neuron_coverage_000)
                                covered_neurons, total_neurons, neuron_coverage_020 = eval_nc(model, np_adversaries, 0.20)
                                print('neuron_coverage_020:', neuron_coverage_020)
                                covered_neurons, total_neurons, neuron_coverage_050 = eval_nc(model, np_adversaries, 0.50)
                                print('neuron_coverage_050:', neuron_coverage_050)
                                covered_neurons, total_neurons, neuron_coverage_075 = eval_nc(model, np_adversaries, 0.75)
                                print('neuron_coverage_075:', neuron_coverage_075)
                                
                                adversaries = torch.transpose(torch.squeeze(torch.tensor(np.stack(np_adversaries))), 3, 1).float().to(device)

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

                                # output diversity
                                pert_output = model.predict(np_adversaries[...,:2]) # model(adversaries)
                                pert_output = torch.tensor(pert_output).to(device) # convert back to torch tensor
                                y_pred = discretize(pert_output, dataset.boundaries).view(-1)

                                output_bias, y_pred_entropy, max_entropy = calculate_output_bias(classes, y_pred)
                                
                                out = {'desc': 'PGD adversaries with diversity regularization.', 
                                       'timestamp': timestamp, 
                                       'attack': attack.__name__, 
                                       'model': model_name, 
                                       'layer': in_out_path, 
                                       'epsilon': e,
                                       'regularization_weight': rw, 
                                       'mse': mse,
                                       'pert_acc': pert_acc, 
                                       'orig_acc': orig_acc,
                                       'attack_success_rate': attack_success_rate,
                                       'neuron_coverage_000': neuron_coverage_000,
                                       'neuron_coverage_020': neuron_coverage_020,
                                       'neuron_coverage_050': neuron_coverage_050,
                                       'neuron_coverage_075': neuron_coverage_075,
                                       'inception_score': mean_is,
                                       'fid_score_64': fid_score_64,
                                       'fid_score_2048': fid_score_2048,
                                       'output_bias': output_bias}
                                
                                results.append(out)
                            
                                # save incremental outputs
                                with open(save_file_path, 'wb') as handle:
                                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                            except Exception as e: 

                                print(str(traceback.format_exc()))
                                error_log.write("Failed on attack_detail {0}: {1}\n".format(str(attack_detail), str(traceback.format_exc())))

                            finally:

                                pass


if __name__ == '__main__':
    try:
        main()
    except Exception as e: 
        print(traceback.format_exc())