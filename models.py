import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import datetime
import glob
import os

class DiversityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_sizes = {}
        self.covered_dict = {} 
        self.lowhigh_dict = {}
        self.hooks = []

    # def track_outputs(self):  
    #     self.outputs = []  
    #     def hook(module, input, output):
    #         self.outputs.append(output.cpu())
    #     for name, module in self.named_children():
    #         if (not isinstance(module, nn.Sequential)
    #             and not isinstance(module, nn.ModuleList)
    #             and not (module == self)):
    #             self.hooks.append(module.register_forward_hook(hook))   

    def extract_outputs(self, data, layer, neuron=None):
        outputs = []      
        def hook(module, input, output):
            outputs.append(output)    
        for name, module in self.named_children():
            if name == layer:
                handle = module.register_forward_hook(hook)     
        out = self(data)
        if not neuron is None:
            outputs[0] = outputs[0][0][neuron]
        else:
            outputs[0] = outputs[0][0]
        handle.remove()
        return torch.stack(outputs)

    def set_layer_output_sizes(self, data): 
        layers = list(dict(self.named_children()))   
        def hook(module, input, output):
            module_idx = len(self.output_sizes)
            m_key = (layers[module_idx], module)
            self.output_sizes[m_key] = list(output.size()[1:])      
        for name, module in self.named_children():
            if (not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == self)):
                self.hooks.append(module.register_forward_hook(hook))
        try:
            self(data[:1])  
        finally:
            self.remove_hooks()

    def init_dict(self, dict, data, tartget_layer, init_value=False): 
        if not self.output_sizes:
            self.set_layer_output_sizes(data)
        for (layer, module), output_size in self.output_sizes.items():
            for index in range(np.prod(output_size)):
                try:
                    val = init_value.copy()
                except:
                    val = init_value
                if tartget_layer is not None:
                    if tartget_layer.lower() in layer.lower():
                        dict[(layer, index)] = val
                else:
                    dict[(layer, index)] = val             

    def neurons_covered(self):
        covered_neurons = len([v for v in self.covered_dict.values() if v])
        total_neurons = len(self.covered_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def scale(out, rmax=1, rmin=0):
        output_std = (out - out.min()) / (out.max() - out.min())
        output_scaled = output_std * (rmax - rmin) + rmin
        return output_scaled

    def update_coverage(self, data, threshold=0.):   
        layers = list(set([x[0] for x in list(self.covered_dict.keys())]))    
        for layer in layers:       
            outputs = torch.squeeze(self.extract_outputs(data, layer))
            scaled_outputs = scale(outputs)            
            for i, out in enumerate(scaled_outputs.view(-1)):
                if out > threshold:
                    covered_dict[(layer, i)] = True
                    
    def eval_nc(self, data, threshold=0.):
        covered_dict = self.init_dict(data, False)
        self.update_coverage(data, threshold=0.)
        _, _, nc = neurons_covered()
        return nc

    def hook_lowhigh_dict(self, layer_name=None):

        def hook(module, input, output):

            output_min = output.min(dim=0).values.reshape(-1)
            output_max = output.max(dim=0).values.reshape(-1)

            # print(output_max.size(), output_min.size())

            for val in list(self.output_sizes):
                if module == val[1]:
                    layer_name = val[0]
                    break

            lowhigh_layer_dict = {k:v for k,v in self.lowhigh_dict.items() if layer_name in k[0]}

            for i, key in enumerate(lowhigh_layer_dict):
                current_lo = self.lowhigh_dict[key]['low']
                current_hi = self.lowhigh_dict[key]['high']
                new_lo = output_min[i].item()
                new_hi = output_max[i].item()
                # print(i, key, current_lo, current_hi, new_lo, new_hi)
                if new_lo < current_lo:
                    self.lowhigh_dict[key]['low'] = new_lo
                if new_hi > current_hi:
                    self.lowhigh_dict[key]['high'] = new_hi

            # # check to see what's going on with the first neuron in the first relu layer
            # print(self.lowhigh_dict.get(('relu1', 0)))

        for name, module in self.named_children():
            if layer_name is not None:
                if (not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == self)
                    and layer_name in name):
                    self.hooks.append(module.register_forward_hook(hook))
            else:
                if (not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == self)):
                    self.hooks.append(module.register_forward_hook(hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()   

class FCNet5(DiversityModel):
    def __init__(self):
        super().__init__()
        self.dens1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.dens2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.dens3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)
        self.dens4 = nn.Linear(64, 20)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.2)
        self.dens5 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dens1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.dens2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.dens3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        x = self.dens4(x)
        x = self.relu4(x)
        x = self.drop4(x)
        x = self.dens5(x)
        return F.log_softmax(x, dim=1)

class FCNet10(DiversityModel):
    def __init__(self):
        super().__init__()
        self.dens1 = nn.Linear(784, 698)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.dens2 = nn.Linear(698, 612)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.dens3 = nn.Linear(612, 526)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)
        self.dens4 = nn.Linear(526, 440)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.2)
        self.dens5 = nn.Linear(440, 354)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.2)
        self.dens6 = nn.Linear(354, 268)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.2)
        self.dens7 = nn.Linear(268, 182)
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout(0.2)
        self.dens8 = nn.Linear(182, 96)
        self.relu8 = nn.ReLU()
        self.drop8 = nn.Dropout(0.2)
        self.dens9 = nn.Linear(96, 20)
        self.relu9 = nn.ReLU()
        self.drop9 = nn.Dropout(0.2)
        self.dens10 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dens1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.dens2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.dens3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        x = self.dens4(x)
        x = self.relu4(x)
        x = self.drop4(x)
        x = self.dens5(x)
        x = self.relu5(x)
        x = self.drop5(x)
        x = self.dens6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.dens7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        x = self.dens8(x)
        x = self.relu8(x)
        x = self.drop8(x)
        x = self.dens9(x)
        x = self.relu9(x)
        x = self.drop9(x)
        x = self.dens10(x)
        return F.log_softmax(x, dim=1)

class Conv2DNet(DiversityModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

class Conv1DNet(DiversityModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 20, 5, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(20, 50, 5, 1)
        self.relu2 = nn.ReLU()
        self.dens1 = nn.Linear(193 * 50, 500)
        self.relu3 = nn.ReLU()
        self.dens2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1).unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool1d(x, 2, 2)
        x = x.view(-1, 193 * 50)
        x = self.dens1(x)
        x = self.relu3(x)
        x = self.dens2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # calculate robust loss
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            batch_size = len(data)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc


def get_dict_for_layer(dict, layer_name):
    return {k:v for k,v in dict.items() if layer_name in k[0]}

def get_pretrained_weights(model, directory="pretrained_models/mnist/"):

    latest_model = None
    m_type = model.__class__.__name__
    prev_models = glob.glob(directory+'*'+ m_type +'*.pth')
    if prev_models:
        latest_model = max(prev_models, key=os.path.getctime)

    if (latest_model is not None 
        and m_type in latest_model):  
        print('loading model', latest_model)
        model.load_state_dict(torch.load(latest_model))  
    else:
        print('no model found. train a new one.')

    return model
