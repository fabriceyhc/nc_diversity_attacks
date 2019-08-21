import torch
import torchvision
import torchvision.transforms as transforms

batch_size_train = 64
batch_size_test = 100

transform = transforms.Compose([
 transforms.Resize(256),        
 transforms.CenterCrop(224),    
 transforms.ToTensor(),         
 transforms.Normalize(          
 mean=[0.485, 0.456, 0.406],    
 std=[0.229, 0.224, 0.225]      
 )])

data_dir = 'C:\data\ImageNet'

train_set = torchvision.datasets.ImageNet(root=data_dir, split='train', download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=2)

test_set = torchvision.datasets.ImageNet(root=data_dir, split='val', download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False, num_workers=2)