
from torch import nn
from torch.nn import functional as F
import torch
from torch.nn import init

z_dim1, z_dim2 = 8, 16
inner_1 = 32
inner_2 = 128
inner_21, inner_22 = 128, 16
z_dim = z_dim1 + z_dim2 + 45 + 65
a = nn.Dropout(p = 0.5)  
class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.f1, self.f2 = nn.Linear(65, inner_1, bias = False), nn.Linear(inner_1, z_dim1, bias = False)
        self.f3, self.f4 = nn.Linear(z_dim1, inner_1, bias = False), nn.Linear(inner_1, 65, bias = False)
        self.relu = nn.ReLU(inplace = True)
    def encode(self, x):
        x = self.f2(self.relu(self.f1(x)))
        return x

    def decode(self, z):
        z = self.f4(self.relu(self.f3(z)))
        return torch.sigmoid(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return z, recon_x


class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.f1, self.f2 = nn.Linear(292, inner_2, bias = False), nn.Linear(inner_2, z_dim2, bias = False)
        self.f3, self.f4 = nn.Linear(z_dim2, inner_2, bias = False), nn.Linear(inner_2, 292, bias = False)
        self.relu = nn.ReLU(inplace = True)
    def encode(self, x):
        x = self.f2(self.relu(self.f1(x)))
        return x

    def decode(self, z):
        z = self.f4(self.relu(self.f3(z)))
        return torch.sigmoid(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return z, recon_x


    
class classD(nn.Module):
    def __init__(self):
        super(classD, self).__init__()
        self.f1, self.f2 = nn.Linear(z_dim, int(z_dim/2), bias = False), nn.Linear(int(z_dim/2), 2, bias = False)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        z = self.f2(self.relu(self.f1(x)))
        #z = self.f2(a(self.f1(x)))
        #z = self.f2(self.f1(x))
        return z
        
