import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from utils import dict_drop

def get_res_net_20(ensemble, network_hyperparams):
    if ensemble is None:
        return ResNet20(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type","dropout","dropout_exit", "dropout_p","n_exits", "exit_after",
            "complexity_factor"))
            
    elif ensemble == "early_exit":
        return ResNet20EarlyExit(**dict_drop(network_hyperparams,"call", "load_model","resnet_type","dropout","dropout_exit", "dropout_p", "n_exits"))

    elif ensemble == "mc":
        return ResNet20MCDrop(**dict_drop(network_hyperparams,"call", "load_model","resnet_type", "n_exits","exit_after"))

    elif ensemble == "mc_early_exit":
        return ResNet20MCEarlyExit(**dict_drop(network_hyperparams,"call", "load_model","resnet_type", "n_exits"))

# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
        

class ShortCutLayer(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.planes = planes

    def forward(self,x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes//4, self.planes//4), "constant", 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = ShortCutLayer(planes)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ExitBlock(nn.Module):

    def __init__(self, in_channels, hidden_sizes, out_channels):
        super().__init__()

        layers = [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Flatten(1)]
        layers += [nn.Linear(in_channels, hidden_sizes)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_sizes, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class MCExitBlock(nn.Module):
    def __init__(self, in_channels, hidden_sizes, out_channels, p_dropout):
        super().__init__()

        layers = [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Flatten(1)]
        layers += [nn.Linear(in_channels, hidden_sizes)]
        layers += [nn.ReLU()]
        layers += [MCDropout(p_dropout)]
        layers += [nn.Linear(hidden_sizes, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, out_dim=100):
        super().__init__()
        self.out_dim = out_dim
        self.in_planes = 16
        self.hidden_sizes = [16,32,64]
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(16)
        relu1 = nn.ReLU(inplace=True)
        in_block = [conv1,bn1,relu1]
        self.in_block = nn.Sequential(*in_block)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        blocks = [self.layer1,self.layer2,self.layer3]
        self.blocks = nn.Sequential(*blocks)

        out_block = [nn.AdaptiveAvgPool2d(1)]
        out_block += [nn.Flatten(1)]
        out_block += [nn.Linear(64, out_dim)]
        self.out_block = nn.Sequential(*out_block)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.in_block(x)
        x = self.blocks(x)
        out = self.out_block(x)
        return [out]


class ResNet20(ResNet):
    def __init__(self,*args,  **kwargs):
        super().__init__(BasicBlock, [3, 3, 3], *args,  **kwargs)
        self.n_exits = 1

# https://github.com/ajrcampbell/early-exit-ensembles/blob/main/src/models/res_net_18.py
class ResNet20EarlyExit(ResNet20):
    
    name = "res_net_20_early_exit"

    def __init__(self, exit_after=-1, complexity_factor=1.2, *args,  **kwargs):
        self.exit_after = exit_after
        self.complexity_factor = complexity_factor
        super().__init__(*args, **kwargs)
        to_exit = [2, 8, 14, 21, 27, 33, 40, 46, 52]
        hidden_sizes = len(self.hidden_sizes)

        num_hidden = len(self.hidden_sizes)
        exit_hidden_sizes = [int(((self.complexity_factor ** 0.5) ** (num_hidden - idx)) * self.hidden_sizes[-1]) for idx in range(num_hidden)]
        # Have 3 blocks for each hidden size in ResNet-20
        exit_hidden_sizes = [h for pair in zip(exit_hidden_sizes, exit_hidden_sizes,exit_hidden_sizes) for h in pair]
        if self.exit_after == -1:
            self.exit_after = range(len(to_exit))

        early_exits = len(to_exit)
        self.n_exits = early_exits + 1

        if (len(self.exit_after) > early_exits) or not set(self.exit_after).issubset(list(range(early_exits))):
            raise ValueError("valid exit points: {}".format(", ".join(str(n) for n in range(early_exits))))
        self.exit_hidden_sizes = np.array(exit_hidden_sizes)[self.exit_after]

        blocks = []
        for idx, module in enumerate(self.blocks.modules()):
            if idx in to_exit:
                blocks += [module]
        self.blocks = nn.ModuleList(blocks)

        idx = 0
        exit_blocks = []
        for block_idx, block in enumerate(self.blocks):
            if block_idx in self.exit_after:
                in_channels = block.conv1.out_channels
                exit_blocks += [ExitBlock(in_channels, self.exit_hidden_sizes[idx], self.out_dim)]
                idx += 1
        self.exit_blocks = nn.ModuleList(exit_blocks)

        self.apply(_weights_init)

    def forward(self, x):

        out = self.in_block(x)

        out_blocks = []
        for block in self.blocks:
            out = block(out)
            out_blocks += [out]

        out_exits = []
        for exit_after, exit_block in zip(self.exit_after, self.exit_blocks):
            out = exit_block(out_blocks[exit_after])
            out_exits += [out]

        out = self.out_block(out_blocks[-1])
        #out = torch.stack(out_exits + [out], dim=0)
        out_exits += [out]
        return out_exits

class ResNet20MCEarlyExit(ResNet20):
    
    name = "res_net_20_mc_early_exit"

    def __init__(self, dropout = None, dropout_exit = False, dropout_p = 0.25, exit_after=-1, complexity_factor=1.2, *args,  **kwargs):
        self.dropout_p = dropout_p
        self.dropout = dropout
        self.dropout_exit = dropout_exit
        self.exit_after = exit_after
        self.complexity_factor = complexity_factor
        super().__init__(*args, **kwargs)
        to_exit = [2, 8, 14, 21, 27, 33, 40, 46, 52]
        hidden_sizes = len(self.hidden_sizes)

        num_hidden = len(self.hidden_sizes)
        exit_hidden_sizes = [int(((self.complexity_factor ** 0.5) ** (num_hidden - idx)) * self.hidden_sizes[-1]) for idx in range(num_hidden)]
        # Have 3 blocks for each hidden size in ResNet-20
        exit_hidden_sizes = [h for pair in zip(exit_hidden_sizes, exit_hidden_sizes,exit_hidden_sizes) for h in pair]
        if self.exit_after == -1:
            self.exit_after = range(len(to_exit))

        early_exits = len(to_exit)
        self.n_exits = early_exits + 1

        if (len(self.exit_after) > early_exits) or not set(self.exit_after).issubset(list(range(early_exits))):
            raise ValueError("valid exit points: {}".format(", ".join(str(n) for n in range(early_exits))))
        self.exit_hidden_sizes = np.array(exit_hidden_sizes)[self.exit_after]

        blocks = []
        for idx, module in enumerate(self.blocks.modules()):
            if idx in to_exit:
                blocks += [module]
        self.blocks = nn.ModuleList(blocks)

        if self.dropout is not None:
            for block_idx in self.exit_after:            
                self.blocks[block_idx].add_module("dropout", MCDropout(self.dropout_p))

        idx = 0
        exit_blocks = []
        for block_idx, block in enumerate(self.blocks):
            if block_idx in self.exit_after:
                in_channels = block.conv1.out_channels
                if self.dropout_exit:
                    exit_blocks += [MCExitBlock(in_channels, self.exit_hidden_sizes[idx], self.out_dim, self.dropout_p)]
                else:
                    exit_blocks += [ExitBlock(in_channels, self.exit_hidden_sizes[idx], self.out_dim)]
                idx += 1
        self.exit_blocks = nn.ModuleList(exit_blocks)

        self.apply(_weights_init)

    def forward(self, x):

        out = self.in_block(x)

        out_blocks = []
        for block in self.blocks:
            out = block(out)
            out_blocks += [out]

        out_exits = []
        for exit_after, exit_block in zip(self.exit_after, self.exit_blocks):
            out = exit_block(out_blocks[exit_after])
            out_exits += [out]

        out = self.out_block(out_blocks[-1])
        #out = torch.stack(out_exits + [out], dim=0)
        out_exits += [out]
        return out_exits


class MCDropout(nn.Dropout):

    def forward(self, x):
        return F.dropout(x, self.p, True, self.inplace)

class ResNet20MCDrop(ResNet20EarlyExit):
    
    name = "res_net_20_mc_drop"

    def __init__(self, dropout_p=0.2, dropout = None, dropout_exit = False, *args, **kwargs):
        self.drop_after = -1
        self.drop_prob = dropout_p
        self.dropout = dropout
        self.dropout_exit = dropout_exit

        super().__init__(exit_after=self.drop_after, *args, **kwargs)

        self.drop_after = self.exit_after

        self.__delattr__("exit_after")
        self.__delattr__("exit_blocks")
        if self.dropout is not None:
            for block_idx in self.drop_after:            
                self.blocks[block_idx].add_module("dropout", MCDropout(self.drop_prob))
        
        self.blocks = nn.Sequential(*self.blocks)

        if self.dropout_exit:
            self.__delattr__("out_block")
            out_block = [nn.AdaptiveAvgPool2d(1)]
            out_block += [nn.Flatten(1)]
            out_block += [MCDropout(self.drop_prob)]
            out_block += [nn.Linear(64, self.out_dim)]
            self.out_block = nn.Sequential(*out_block)

    def forward(self, x):

        x = self.in_block(x)
        x = self.blocks(x)
        x = self.out_block(x)
        return [x]

if __name__ == "__main__":
    resnet_ee = ResNet20EarlyExit()
    resnet_mc = ResNet20MCDrop()


    
