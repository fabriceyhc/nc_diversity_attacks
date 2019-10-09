import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

import matplotlib.pyplot as plt

import faulthandler
faulthandler.enable()

# provides a nice UI element when running in a notebook, otherwise use "import tqdm" only
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

def cw_attack(model, inputs, targets, device, targeted=False, norm_type='inf',  
              epsilon=100., confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, 
              max_steps=1000, abort_early=True, box=(-1., 1.), optimizer_lr=1e-2,
              init_rand=False, log_frequency=10):

    batch_size = inputs.size(0)
    num_classes = model(torch.tensor(inputs[0][None,:], requires_grad=False)).size(1)

    ae_tol = torch.tensor(1e-4, device=device)

    # `lower_bounds`, `upper_bounds` and `scale_consts` are used
    # for binary search of each `scale_const` in the batch. The element-wise
    # inquality holds: lower_bounds < scale_consts <= upper_bounds
    lower_bounds = torch.tensor(torch.zeros(batch_size), dtype=torch.float, device=device)
    upper_bounds = torch.tensor(torch.ones(batch_size) * c_range[1], dtype=torch.float, device=device)
    scale_consts = torch.tensor(torch.ones(batch_size) * c_range[0], dtype=torch.float, device=device)

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)
    o_best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    for const_step in range(search_steps):

        print('Step', const_step)

        # the minimum norms of perturbations found during optimization
        best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)

        # the perturbed predictions made by the adversarial perturbations with the smallest norms
        best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)

        # previous (summed) batch loss, to be used in early stopping policy
        prev_batch_loss = torch.tensor(1e10, device=device)

        # optimization steps
        for optim_step in range(max_steps):

            adversaries = from_tanh_space(inputs_tanh + pert_tanh)
            pert_outputs = model(adversaries)
            
            if norm_type == 'inf':
                inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
                norms = inf_norms
            elif norm_type == 'l2':
                l2_norms = torch.pow(adversaries - inputs, exponent=2)
                l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
                norms = l2_norms
            else:
                raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 
                
            target_activ = torch.sum(targets_oh * pert_outputs, 1)
            maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]

            if targeted:           
                # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
                f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
            else:
                # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
                f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)

            # the total loss of current batch, should be of dimension [1]
            cw_loss = torch.sum(scale_consts * f)
            norm_loss = torch.sum(norms)
            batch_loss = cw_loss + norm_loss

            # Do optimization for one step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # "returns" batch_loss, pert_norms, pert_outputs, adversaries

            if optim_step % log_frequency == 0: 
                print('batch [{}] batch_loss: {} cw_loss: {} norm_loss: {}'.format(optim_step, batch_loss, cw_loss, norm_loss))
                
                # print(o_best_norm)
            if abort_early and not optim_step % (max_steps // 10):   
                if batch_loss > prev_batch_loss * (1 - ae_tol):
                    break
                if batch_loss == 0:
                    break
                prev_batch_loss = batch_loss

            # update best attack found during optimization
            pert_predictions = torch.argmax(pert_outputs, dim=1)
            comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)
            for i in range(batch_size):
                norm = norms[i]
                cppred = comp_pert_predictions[i]
                ppred = pert_predictions[i]
                tlabel = targets[i]
                ax = adversaries[i] 
                if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                    assert cppred == ppred
                    if norm < best_norm[i]:
                        best_norm[i] = norm
                        best_norm_ppred[i] = ppred
                    if norm < o_best_norm[i]:
                        o_best_norm[i] = norm
                        o_best_norm_ppred[i] = ppred
                        o_best_adversaries[i] = ax

        # binary search of `scale_const`
        for i in range(batch_size):
            tlabel = targets[i]
            if best_norm_ppred[i] != -1:
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
                    
    return o_best_adversaries

def cw_div1_attack(model, modules, regularizer_weight, inputs, targets, device, targeted=False, norm_type='inf', epsilon=100.,
                   confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                   abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                   init_rand=False, log_frequency=10):

    batch_size = inputs.size(0)
    num_classes = model(torch.tensor(inputs[0][None,:], requires_grad=False)).size(1)

    # `lower_bounds`, `upper_bounds` and `scale_consts` are used
    # for binary search of each `scale_const` in the batch. The element-wise
    # inquality holds: lower_bounds < scale_consts <= upper_bounds
    lower_bounds = torch.tensor(torch.zeros(batch_size), dtype=torch.float, device=device)
    upper_bounds = torch.tensor(torch.ones(batch_size) * c_range[1], dtype=torch.float, device=device)
    scale_consts = torch.tensor(torch.ones(batch_size) * c_range[0], dtype=torch.float, device=device)

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)
    o_best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    for const_step in range(search_steps):

        print('Step', const_step)

        # the minimum norms of perturbations found during optimization
        best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)

        # the perturbed predictions made by the adversarial perturbations with the smallest norms
        best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)

        # previous (summed) batch loss, to be used in early stopping policy
        prev_batch_loss = torch.tensor(1e10, device=device)
        ae_tol = torch.tensor(1e-4, device=device)

        # optimization steps
        for optim_step in range(max_steps):

            adversaries = from_tanh_space(inputs_tanh + pert_tanh)
            pert_outputs = model(adversaries)
            
            if norm_type == 'inf':
                inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
                norms = inf_norms
            elif norm_type == 'l2':
                l2_norms = torch.pow(adversaries - inputs, exponent=2)
                l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
                norms = l2_norms
            else:
                raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 

            # calculate kl divergence for each input to use for adversary selection
            divs = []
            for i in range(batch_size):
                divs.append(norm_divergence_by_module(data=adversaries[i].unsqueeze(0), model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)) 
            div_norms = torch.tensor(torch.stack(divs), device=device)
            
            # calculate kl divergence for batch to use in loss function
            div_reg = norm_divergence_by_module(data=adversaries, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)

            target_activ = torch.sum(targets_oh * pert_outputs, 1)
            maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]

            if targeted:           
                # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
                f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
            else:
                # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
                f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)

            # the total loss of current batch, should be of dimension [1]
            cw_loss = torch.sum(scale_consts * f)
            norm_loss = torch.sum(norms)
            
            batch_loss = cw_loss + norm_loss + div_reg

            # Do optimization for one step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # "returns" batch_loss, pert_norms, pert_outputs, adversaries

            if optim_step % log_frequency == 0: 
                print('batch [{}] batch_loss: {} cw_loss: {} norm_loss: {} div_reg: {}'.format(optim_step, batch_loss, cw_loss, norm_loss, div_reg))
                # print(o_best_norm)

            if abort_early and not optim_step % (max_steps // 10):   
                if batch_loss > prev_batch_loss * (1 - ae_tol):
                    break
                if batch_loss == 0:
                    break
                prev_batch_loss = batch_loss

            # update best attack found during optimization
            pert_predictions = torch.argmax(pert_outputs, dim=1)
            comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)
            for i in range(batch_size):
                norm = norms[i]
                cppred = comp_pert_predictions[i]
                ppred = pert_predictions[i]
                tlabel = targets[i]
                ax = adversaries[i]
                if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                    assert cppred == ppred
                    if norm < best_norm[i]:
                        best_norm[i] = norm
                        best_norm_ppred[i] = ppred
                    if norm < o_best_norm[i]:
                        o_best_norm[i] = norm
                        o_best_norm_ppred[i] = ppred
                        o_best_adversaries[i] = ax

        # binary search of `scale_const`
        for i in range(batch_size):
            tlabel = targets[i]
            if best_norm_ppred[i] != -1:
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
                    
    return o_best_adversaries #, norms

def cw_div2_attack(model, modules, regularizer_weight, inputs, targets, device, targeted=False, norm_type='inf', epsilon=100.,
                   confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                   abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                   init_rand=False, log_frequency=10):

    batch_size = inputs.size(0)
    num_classes = model(torch.tensor(inputs[0][None,:], requires_grad=False)).size(1)

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)
    o_best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    # previous (summed) batch loss, to be used in early stopping policy
    prev_batch_loss = torch.tensor(1e10, device=device)
    ae_tol = torch.tensor(1e-4, device=device)

    # optimization steps
    for optim_step in range(max_steps):

        adversaries = from_tanh_space(inputs_tanh + pert_tanh)
        pert_outputs = model(adversaries)
        
        if norm_type == 'inf':
            inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
            norms = inf_norms
        elif norm_type == 'l2':
            l2_norms = torch.pow(adversaries - inputs, exponent=2)
            l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
            norms = l2_norms
        else:
            raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 

        # calculate kl divergence for each input to use for adversary selection
        divs = []
        for i in range(batch_size):
            divs.append(norm_divergence_by_module(data=adversaries[i].unsqueeze(0), model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)) 
        div_norms = torch.tensor(torch.stack(divs), device=device)

        # calculate kl divergence for batch to use in loss function
        div_reg = norm_divergence_by_module(data=adversaries, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)

        target_activ = torch.sum(targets_oh * pert_outputs, 1)
        maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]

        if targeted:           
            # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
            f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
        else:
            # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
            f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)

        # the total loss of current batch, should be of dimension [1]
        cw_loss = torch.sum(f)
        norm_loss = torch.sum(norms)
        
        batch_loss = cw_loss + norm_loss + div_reg

        # Do optimization for one step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # "returns" batch_loss, pert_norms, pert_outputs, adversaries

        if optim_step % log_frequency == 0: 
            print('batch [{}] batch_loss: {} cw_loss: {} norm_loss: {} div_reg: {}'.format(optim_step, batch_loss, cw_loss, norm_loss, div_reg))
            # print(o_best_norm)

        if abort_early and not optim_step % (max_steps // 10):   
            if batch_loss > prev_batch_loss * (1 - ae_tol):
                break
            if batch_loss == 0:
                break
            prev_batch_loss = batch_loss

        # update best attack found during optimization
        pert_predictions = torch.argmax(pert_outputs, dim=1)
        comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)
        for i in range(batch_size):
            norm = norms[i]
            cppred = comp_pert_predictions[i]
            ppred = pert_predictions[i]
            tlabel = targets[i]
            ax = adversaries[i]
            if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                assert cppred == ppred
                if norm < o_best_norm[i]:
                    o_best_norm[i] = norm
                    o_best_norm_ppred[i] = ppred
                    o_best_adversaries[i] = ax
                    
    return o_best_adversaries #, norms

def cw_div3_attack(model, modules, regularizer_weight, inputs, targets, device, targeted=False, norm_type='inf', epsilon=100., 
                   confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                   abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                   init_rand=False, log_frequency=10):

    batch_size = inputs.size(0)
    num_classes = model(torch.tensor(inputs[0][None,:], requires_grad=False)).size(1)

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)
    o_best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    # previous (summed) batch loss, to be used in early stopping policy
    prev_batch_loss = torch.tensor(1e10, device=device)
    ae_tol = torch.tensor(1e-4, device=device)

    # optimization steps
    for optim_step in range(max_steps):

        adversaries = from_tanh_space(inputs_tanh + pert_tanh)
        pert_outputs = model(adversaries)
        
        if norm_type == 'inf':
            inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
            norms = inf_norms
        elif norm_type == 'l2':
            l2_norms = torch.pow(adversaries - inputs, exponent=2)
            l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
            norms = l2_norms
        else:
            raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 

        # calculate kl divergence for each input to use for adversary selection
        divs = []
        for i in range(batch_size):
            divs.append(norm_divergence_by_module(data=adversaries[i].unsqueeze(0), model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)) 
        div_norms = torch.tensor(torch.stack(divs), device=device)

        # calculate kl divergence for batch to use in loss function
        div_reg = norm_divergence_by_module(data=adversaries, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)

        loss = -1. * nn.CrossEntropyLoss()(pert_outputs, targets)

        # the total loss of current batch, should be of dimension [1]
        ce_loss = torch.sum(loss)
        norm_loss = torch.sum(norms)
        
        batch_loss = ce_loss + norm_loss + div_reg

        # Do optimization for one step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # "returns" batch_loss, pert_norms, pert_outputs, adversaries

        if optim_step % log_frequency == 0: 
            print('batch [{}] batch_loss: {} ce_loss: {} norm_loss: {} div_reg: {}'.format(optim_step, batch_loss, ce_loss, norm_loss, div_reg))
            # print(o_best_norm)

        if abort_early and not optim_step % (max_steps // 10):   
            if batch_loss > prev_batch_loss * (1 - ae_tol):
                break
            if batch_loss == 0:
                break
            prev_batch_loss = batch_loss

        # update best attack found during optimization
        pert_predictions = torch.argmax(pert_outputs, dim=1)
        comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)
        for i in range(batch_size):
            norm = norms[i]
            cppred = comp_pert_predictions[i]
            ppred = pert_predictions[i]
            tlabel = targets[i]
            ax = adversaries[i]
            if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                assert cppred == ppred
                if norm < o_best_norm[i]:
                    o_best_norm[i] = norm
                    o_best_norm_ppred[i] = ppred
                    o_best_adversaries[i] = ax
                    
    return o_best_adversaries #, norms

def cw_div4_attack(model, modules, regularizer_weight, inputs, targets, device, targeted=False, norm_type='inf', epsilon=100., 
                   confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                   abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                   init_rand=False, log_frequency=10):

    inputs = inputs.to(device)
    targets = targets.to(device)
    model.to(device)

    batch_size = inputs.size(0)
    with torch.no_grad():
        num_classes = model(inputs[0].unsqueeze(0)).size(1)

    # `lower_bounds`, `upper_bounds` and `scale_consts` are used
    # for binary search of each `scale_const` in the batch. The element-wise
    # inquality holds: lower_bounds < scale_consts <= upper_bounds
    lower_bounds = torch.zeros(batch_size).to(device) 
    upper_bounds = torch.ones(batch_size).to(device) * c_range[1]
    scale_consts = torch.ones(batch_size).to(device) * c_range[0]

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.ones(batch_size).to(device) * 1e10
    o_best_norm_ppred = torch.ones(batch_size).to(device) * -1.
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    for const_step in range(search_steps):

        print('Step', const_step)

        # # the minimum norms of perturbations found during optimization
        # best_norm = torch.ones(batch_size).to(device) * 1e10

        # # the perturbed predictions made by the adversarial perturbations with the smallest norms
        # best_norm_ppred = torch.ones(batch_size).to(device)  * -1.

        # previous (summed) batch loss, to be used in early stopping policy
        prev_batch_loss = torch.tensor(1e10).to(device)
        ae_tol = torch.tensor(1e-4).to(device) # abort early tolerance

        # optimization steps
        for optim_step in range(max_steps):

            adversaries = from_tanh_space(inputs_tanh + pert_tanh)
            pert_outputs = model(adversaries)

            if norm_type == 'inf':
                inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
                norms = inf_norms
            elif norm_type == 'l2':
                l2_norms = torch.pow(adversaries - inputs, exponent=2)
                l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
                norms = l2_norms
            else:
                raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 
            
            # calculate kl divergence for batch to use in loss function
            div_reg = 0
            if regularizer_weight > 0:
                div_reg = norm_divergence_by_module(data=adversaries, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)

            target_activ = torch.sum(targets_oh * pert_outputs, 1)
            maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]

            if targeted:           
                # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
                f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
            else:
                # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
                f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)

            cw_loss = torch.sum(scale_consts * f)
            norm_loss = torch.sum(norms)
            
            batch_loss = cw_loss + norm_loss + div_reg

            # Do optimization for one step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # "returns" batch_loss, pert_norms, pert_outputs, adversaries

            if optim_step % log_frequency == 0: 
                print('batch [{}] batch_loss: {} cw_loss: {} norm_loss: {} div_reg: {}'.format(optim_step, batch_loss, cw_loss, norm_loss, div_reg))
                # print(o_best_norm)

            if abort_early and not optim_step % (max_steps // 10):
                if batch_loss > prev_batch_loss * (1 - ae_tol):
                    break
                if batch_loss == 0:
                    break
                prev_batch_loss = batch_loss

            # update best attack found during optimization
            pert_predictions = torch.argmax(pert_outputs, dim=1)
            comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)

            for i in range(batch_size):
                norm = norms[i]
                cppred = comp_pert_predictions[i]
                ppred = pert_predictions[i]
                tlabel = targets[i]
                ax = adversaries[i]
                if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                    assert cppred == ppred
                    # if norm < best_norm[i]:
                    #     best_norm[i] = norm
                    #     best_norm_ppred[i] = ppred
                    if norm < o_best_norm[i]:
                        o_best_norm[i] = norm
                        o_best_norm_ppred[i] = ppred
                        o_best_adversaries[i] = ax

        # binary search of `scale_const`
        for i in range(batch_size):
            tlabel = targets[i]
            if o_best_norm_ppred[i] != -1:
            # if best_norm_ppred[i] != -1:
                # print('attack successful')
                # successful: attempt to lower `scale_const` by halving it
                if scale_consts[i] < upper_bounds[i]:
                    upper_bounds[i] = scale_consts[i]
                # `upper_bounds[i] == c_range[1]` implies no solution
                # found, i.e. upper_bounds[i] has never been updated by
                # scale_consts[i] until `scale_consts[i] > 0.1 * c_range[1]`
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
            else:
                # print('attack failed')
                # failure: multiply `scale_const` by ten if no solution
                # found; otherwise do binary search
                if scale_consts[i] > lower_bounds[i]:
                    lower_bounds[i] = scale_consts[i]
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
                else:
                    scale_consts[i] *= 10  

    return o_best_adversaries 

# HELPER FUNCTIONS

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
  
def compensate_confidence(outputs, targets, targeted, confidence):
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
    rng = torch.arange(start=0, end=targets.shape[0], dtype=torch.long)
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
  
def attack_successful(prediction, target, targeted):
    """
    See whether the underlying attack is successful.
    """
    if targeted:
        return prediction == target
    else:
        return prediction != target

def extract_outputs(model, data, module):
    outputs = []      
    def hook(module, input, output):
        outputs.append(output)    
    handle = module.register_forward_hook(hook)     
    model(data)
    handle.remove()
    return torch.stack(outputs)

def norm_divergence_by_module(data, model, modules, device, regularizer_weight=None):
    """
    returns the kld between the activations of the specified layer and a uniform pdf
    """

    if not isinstance(modules, list):
        modules = [modules]

    data = torch.clamp(data, 0, 1)

    total_divergence = 0

    for module in modules: 
    
        # extract layer activations as numpy array
        # NOTE: torch.relu is added just in case the layer is not actually ReLU'd beforehand
        #       This is required for the summation and KL-Divergence calculation, otherwise nan
        layer_activations = torch.relu(torch.squeeze(extract_outputs(model=model, data=data, module=module)))
        
        # normalize over summation (to get a probability density)
        if len(layer_activations.size()) == 1:
            out_norm = (layer_activations / torch.sum(layer_activations)) + 1e-20 
        elif len(layer_activations.size()) == 2:
            out_norm = torch.sum(layer_activations, 0)
            out_norm = (out_norm / torch.sum(out_norm)) + 1e-20
        else:
            out_norm = (layer_activations / torch.sum(layer_activations)) + 1e-20 

        # create uniform tensor
        uniform_tensor = torch.ones(out_norm.shape).to(device)

        # normalize over summation (to get a probability density)
        uni_norm = uniform_tensor / torch.sum(uniform_tensor)
        
        # measure divergence between normalized layer activations and uniform distribution
        divergence = F.kl_div(input=out_norm.log(), target=uni_norm, reduction='sum')
        # divergence = F.kl_div(input=uni_norm.log(), target=out_norm, reduction='sum') 
        
        # default regularizer if not provided
        if regularizer_weight is None:
            regularizer_weight = 0.005 
            
        if divergence < 0:
            print('The divergence was technically less than 0', divergence, layer_activations, out_norm)
            torch.save(data, 'logs/data.pt')
            torch.save(out_norm, 'logs/out_norm.pt')
            torch.save(uni_norm, 'logs/uni_norm.pt')
            # return None

        total_divergence += divergence
    
    return regularizer_weight * total_divergence

def eval_performance(model, originals, adversaries, targets):
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

def sample_1D_images(model, originals, adversaries, targets, num_samples = 5):
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

def sample_3D_images(model, originals, adversaries, targets, classes, num_samples = 5):
    orig_inputs = originals.cpu().detach().numpy()
    adv_examples = adversaries.cpu().detach().numpy()
    pert_output = model(adversaries)
    orig_output = model(originals)
    pert_pred = torch.argmax(pert_output, dim=1)
    orig_pred = torch.argmax(orig_output, dim=1)
    plt.figure(figsize=(15,8))
    for i in range(1, num_samples+1):
        plt.subplot(2, num_samples, i)
        plt.imshow(np.transpose(np.squeeze(orig_inputs[i]), (1, 2, 0)))  
        true_idx = targets[i].item()
        plt.title('true: {}'.format(classes[true_idx]))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, num_samples, num_samples+i)
        plt.imshow(np.transpose(np.squeeze(adv_examples[i]), (1, 2, 0)))  
        pred_idx = pert_pred[i].item()
        orig_idx = orig_pred[i].item()
        plt.title('adv_pred: {} - orig_pred: {}'.format(classes[pred_idx], classes[orig_idx]))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

def generate_batch(dataset, num_per_class, device):
    '''
    creates a batch of inputs with a customizable number of instances for each class
    dataset       : torchvision.dataset
    num_per_class : iterable containing the desired counts of each class
                    example: torch.ones(num_classes) * 100
    '''
    
    def get_same_index(targets, label):
        '''
        Returns indices corresponding to the target label
        which the dataloader uses to serve downstream.
        '''
        label_indices = []
        for i in range(len(targets)):
            if targets[i] == label:
                label_indices.append(i)
        return label_indices

    data = []
    labels = []
    
    num_classes = len(np.unique(dataset.targets))
    
    for i in range(num_classes):
        
        target_indices = get_same_index(dataset.targets, i)
        class_batch_size = int(num_per_class[i])
        
        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=class_batch_size, 
            sampler=SubsetRandomSampler(target_indices),
            shuffle=False,
            pin_memory=True)

        inputs, targets = next(iter(data_loader))

        data.append(inputs)
        labels.append(targets)

    inputs = torch.cat(data, dim=0).to(device)
    targets = torch.cat(labels, dim=0).to(device)
    
    return inputs, targets

def step_through_model(model, prefix=''):
    for name, module in model.named_children():
        path = '{}/{}'.format(prefix, name)
        if (isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)): # test for dataset
            yield (path, name, module)
        else:
            yield from step_through_model(module, path)

def get_model_layers(model):
    layer_dict = {}
    idx=1
    for (path, name, module) in step_through_model(model):
        layer_dict[path + '-' + str(idx)] = module
        idx += 1
    return layer_dict 