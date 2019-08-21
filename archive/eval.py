import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('error')

import datetime
import glob
import os

import pickle

from models import *

# load the results from file
with open('assets/results.pickle', 'rb') as handle:
    results = pickle.load(handle)

n_epochs = 10
batch_size_train = 64
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5

random_seed = 1
torch.manual_seed(random_seed)

# check if CUDA is available
device = torch.device("cpu")
if torch.cuda.is_available():
    print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda")
else:
    print('CUDA is not available.  Training on CPU ...')

#  torchvision.transforms.Normalize(
#    (0.1307,), (0.3081,))

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
    batch_size=batch_size_train, shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=False, download=True,
                         transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                         ])),
    batch_size=batch_size_test, shuffle=True, pin_memory=True)

# targets = true labels only for when you're doing a targeted attack
# otherwise, you're going to make the inputs easier to classify to 
# do a targeted attack, targets should be some class other than
# the true label

inputs, targets = next(iter(test_loader))

inputs = inputs.to(device)
targets = targets.to(device)

retrain = False
track_low_high = False

model = ConvNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# check to see if we can just load a previous model
# %mkdir models
latest_model = None
m_type = model.__class__.__name__
prev_models = glob.glob('pretrained_models/*'+ m_type +'*.pth')
if prev_models:
    latest_model = max(prev_models, key=os.path.getctime)

if (retrain is False 
    and latest_model is not None 
    and m_type in latest_model):  
    print('loading model', latest_model)
    model.load_state_dict(torch.load(latest_model))  
else:
    if track_low_high:
        model.init_dict(model.lowhigh_dict, inputs, 'relu', {'low': 0, 'high': 0})
        try:
            for epoch in range(1, n_epochs + 1):
                model.hook_lowhigh_dict('relu')
                train(model, device, train_loader, optimizer, epoch)
                model.remove_hooks()
                test(model, device, test_loader)    
        finally:
            model.remove_hooks()   
    else:
        for epoch in range(1, n_epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            acc = test(model, device, test_loader)  
    torch.save(model.state_dict(), 'pretrained_models/model_' + m_type + '_' + str(datetime.datetime.now()).replace(':','.') + '_' + str(acc) + '.pth')

def atanh(x, eps=1e-2):
    """
    The inverse hyperbolic tangent function, missing in pytorch.

    :param x: a tensor or a Variable
    :param eps: used to enhance numeric stability
    :return: :math:`\\tanh^{-1}{x}`, of the same type as ``x``
    """
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))

def to_tanh_space(x, box=(-1., 1.)):
    """
    Convert a batch of tensors to tanh-space. This method complements the
    implementation of the change-of-variable trick in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in tanh-space, of the same dimension;
             the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return atanh((x - _box_plus) / _box_mul)

def from_tanh_space(x, box=(-1., 1.)):
    """
    Convert a batch of tensors from tanh-space to oridinary image space.
    This method complements the implementation of the change-of-variable trick
    in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in ordinary image space, of the same
             dimension; the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return torch.tanh(x) * _box_mul + _box_plus
  
def compensate_confidence(outputs, targets):
    """
    Compensate for ``self.confidence`` and returns a new weighted sum
    vector.

    :param outputs: the weighted sum right before the last layer softmax
           normalization, of dimension [B x M]
    :type outputs: np.ndarray
    :param targets: either the attack targets or the real image labels,
           depending on whether or not ``self.targeted``, of dimension [B]
    :type targets: np.ndarray
    :return: the compensated weighted sum of dimension [B x M]
    :rtype: np.ndarray
    """
    outputs_comp = outputs.clone()
    rng = torch.arange(start=0, end=targets.shape[0], device=device)
    # targets = targets.int()
    if targeted:
        # for each image $i$:
        # if targeted, `outputs[i, target_onehot]` should be larger than
        # `max(outputs[i, ~target_onehot])` by `self.confidence`
        outputs_comp[rng, targets] -= confidence
    else:
        # for each image $i$:
        # if not targeted, `max(outputs[i, ~target_onehot]` should be larger
        # than `outputs[i, target_onehot]` (the ground truth image labels)
        # by `self.confidence`
        outputs_comp[rng, targets] += confidence
    return outputs_comp
  
def attack_successful(prediction, target):
    """
    See whether the underlying attack is successful.
    """
    if targeted:
        return prediction == target
    else:
        return prediction != target
      
def norm_divergence(data, model, layer, neuron=None, regularizer_weight=None):
    """
    returns the kld between the activations of the specified layer and a uniform pdf
    """
    # extract layer activations as numpy array
    layer_activations = torch.squeeze(model.extract_outputs(data=data, layer=layer))
    
    # normalize over summation (to get a probability density)
    out_norm = torch.sum(layer_activations, 0)
    out_norm = out_norm / torch.sum(out_norm) + 1e-6 # F.softmax(out_norm, 1)

    # create uniform tensor
    uniform_tensor = torch.ones(out_norm.shape).to(device)

    # normalize over summation (to get a probability density)
    uni_norm = uniform_tensor / torch.sum(uniform_tensor)
    
    # measure divergence between normalized layer activations and uniform distribution
    divergence = F.kl_div(input=out_norm.log(), target=uni_norm, reduction='sum')
    
    # default regularizer if not provided
    if regularizer_weight is None:
        regularizer_weight = 0.005 
    
    return regularizer_weight * divergence

def eval_performance(model, originals, adversaries):
    print('eval_performance')
    pert_output = model(adversaries)
    orig_output = model(originals)

    pert_pred = torch.argmax(pert_output, dim=1)
    orig_pred = torch.argmax(orig_output, dim=1)

    pert_correct = pert_pred.eq(targets.data).sum()
    orig_correct = orig_pred.eq(targets.data).sum()

    pert_acc = 100. * pert_correct / len(targets)
    orig_acc = 100. * orig_correct / len(targets)

    print('Perturbed Accuracy: {}/{} ({:.0f}%)'.format(pert_correct, len(targets), pert_acc))
    print('Original Accuracy: {}/{} ({:.0f}%)'.format(orig_correct, len(targets), orig_acc))
    
    return pert_acc, orig_acc

def sample_images(originals, adversaries, num_samples = 5):
    orig_inputs = originals.cpu().detach().numpy()
    adv_examples = adversaries.cpu().detach().numpy()
    pert_output = model(adversaries)
    orig_output = model(originals)
    pert_pred = torch.argmax(pert_output, dim=1)
    orig_pred = torch.argmax(orig_output, dim=1)
    plt.figure(figsize=(15,8))
    for i in range(1, num_samples+1):
        plt.subplot(2, num_samples, i)
        plt.imshow(np.squeeze(orig_inputs[i]), cmap='gray')  
        plt.title('true: {}'.format(targets[i].item()))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, num_samples, num_samples+i)
        plt.imshow(np.squeeze(adv_examples[i]), cmap='gray')
        plt.title('adv_pred: {} - orig_pred: {}'.format(pert_pred[i].item(), orig_pred[i].item()))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


targeted=False
confidence=0.0
c_range=(1e-3, 1e10)
search_steps=10
max_steps=1000
abort_early=True
optimizer_lr=5e-4

mean = (0.1307,) # the mean used in inputs normalization
std = (0.3081,) # the standard deviation used in inputs normalization
box = (min((0 - m) / s for m, s in zip(mean, std)),
       max((1 - m) / s for m, s in zip(mean, std)))

log_frequency = 100

def cw_l2_attack(model, inputs, targets, targeted=False, confidence=0.0,
                 c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                 abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                 init_rand=False, log_frequency=10):

    batch_size = inputs.size(0)
    num_classes = model(inputs[0][None,:].clone().detach()).size(1)

    # `lower_bounds`, `upper_bounds` and `scale_consts` are used
    # for binary search of each `scale_const` in the batch. The element-wise
    # inquality holds: lower_bounds < scale_consts <= upper_bounds
    lower_bounds = torch.tensor(np.zeros(batch_size), dtype=torch.float, device=device)
    upper_bounds = torch.tensor(np.ones(batch_size) * c_range[1], dtype=torch.float, device=device)
    scale_consts = torch.tensor(np.ones(batch_size) * c_range[0], dtype=torch.float, device=device)

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_l2`          : the least L2 norms
    # - `o_best_l2_ppred`    : the perturbed predictions made by the adversarial perturbations with the least L2 norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_l2_ppred`
    o_best_l2 = torch.tensor(np.ones(batch_size) * np.inf, dtype=torch.float, device=device)
    o_best_l2_ppred = torch.tensor(-np.ones(batch_size), dtype=torch.float, device=device)
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    for const_step in range(search_steps):

        print('Step', const_step)

        # the minimum L2 norms of perturbations found during optimization
        best_l2 = torch.tensor(np.ones(batch_size) * np.inf, dtype=torch.float, device=device)

        print('Step', const_step, 0)

        # the perturbed predictions made by the adversarial perturbations with the least L2 norms
        best_l2_ppred = torch.tensor(-np.ones(batch_size), dtype=torch.float, device=device)

        print('Step', const_step, 1)

        # previous (summed) batch loss, to be used in early stopping policy
        prev_batch_loss = torch.tensor(np.inf, device=device)
        ae_tol = torch.tensor(1e-4, device=device)

        print('Step', const_step, 2)

        print(const_step, 0)

        # optimization steps
        for optim_step in range(max_steps):

            print(const_step, optim_step, 0)

            adversaries = from_tanh_space(inputs_tanh + pert_tanh)
            pert_outputs = model(adversaries)

            print(const_step, optim_step, 1)

            # Calculate L2 norm between adversaries and original inputs
            pert_norms = torch.pow(adversaries - inputs, exponent=2)
            pert_norms = torch.sum(pert_norms.view(pert_norms.size(0), -1), 1)

            target_activ = torch.sum(targets_oh * pert_outputs, 1)
            maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]

            print(const_step, optim_step, 2)

            if targeted:           
                # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
                f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
            else:
                # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
                f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)

            # the total loss of current batch, should be of dimension [1]
            batch_loss = torch.sum(pert_norms + scale_consts * f)

            print(const_step, optim_step, 3)

            # Do optimization for one step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            print(const_step, optim_step, 4)

            # "returns" batch_loss, pert_norms, pert_outputs, adversaries

            if optim_step % log_frequency == 0: 
                print('batch [{}] loss: {}'.format(optim_step, batch_loss))

            if abort_early and not optim_step % (max_steps // 10):   
                if batch_loss > prev_batch_loss * (1 - ae_tol):
                    break
                prev_batch_loss = batch_loss

            print(const_step, optim_step, 5)

            # update best attack found during optimization
            pert_predictions = torch.argmax(pert_outputs, dim=1)
            comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets), dim=1)
            for i in range(batch_size):
                l2 = pert_norms[i]
                cppred = comp_pert_predictions[i]
                ppred = pert_predictions[i]
                tlabel = targets[i]
                ax = adversaries[i]
                if attack_successful(cppred, tlabel):
                    assert cppred == ppred
                    if l2 < best_l2[i]:
                        best_l2[i] = l2
                        best_l2_ppred[i] = ppred
                    if l2 < o_best_l2[i]:
                        o_best_l2[i] = l2
                        o_best_l2_ppred[i] = ppred
                        o_best_adversaries[i] = ax

            print(const_step, optim_step, 6)

        print(const_step, 1)

        # binary search of `scale_const`
        for i in range(batch_size):
            tlabel = targets[i]
            if best_l2_ppred[i] != -1:
                # successful: attempt to lower `scale_const` by halving it
                if scale_consts[i] < upper_bounds[i]:
                    upper_bounds[i] = scale_consts[i]
                # `upper_bounds[i] == c_range[1]` implies no solution
                # found, i.e. upper_bounds[i] has never been updated by
                # scale_consts[i] until `scale_consts[i] > 0.1 * c_range[1]`
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
            else:
                # failure: multiply `scale_const` by ten if no solution
                # found; otherwise do binary search
                if scale_consts[i] > lower_bounds[i]:
                    lower_bounds[i] = scale_consts[i]
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
                else:
                    scale_consts[i] *= 10

        print(const_step, 2)

    print('finished')
                    
    return o_best_adversaries


cw_advs = cw_l2_attack(model, inputs, targets, targeted=False, confidence=0.0,
                       c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                       abort_early=True, box=box, optimizer_lr=5e-4, 
                       init_rand=False, log_frequency=100)

timestamp = str(datetime.datetime.now()).replace(':','.')

print(timestamp, cw_advs.shape)

pert_acc, orig_acc = eval_performance(model, inputs, cw_advs)
sample_images(inputs, cw_advs)

pert_acc = pert_acc.item() / 100.
orig_acc = orig_acc.item() / 100.
            
out = {'timestamp': timestamp, 'attack': 'baseline_cw', 'layer': 'NA', 'regularization_weight': 'NA', 'adversaries': cw_advs, 'divergences':'NA', 'pert_acc':pert_acc, 'orig_acc': orig_acc}
results.append(out)

pickle.dump(results, open( "assets/results2.pickle", "wb" ))