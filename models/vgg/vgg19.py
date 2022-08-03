"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import dict_drop

def get_vgg_19(ensemble, network_hyperparams):
    # if ensemble is None:
    #     # n_exits, out_dim
    #     return VGG19(**dict_drop(network_hyperparams,"call", "load_model",
    #         "resnet_type","dropout","dropout_exit", "dropout_p"))

    if ensemble == "early_exit" or ensemble is None:
        # n_exits, out_dim
        return VGG19EarlyExit(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type","dropout","dropout_exit", "dropout_p"))

    elif ensemble == "mc":
        # n_exits, out_dim, dropout, dropout_exit, dropout_p
        return VGG19MC(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type"))
     
    elif ensemble == "mc_early_exit":
        # n_exits, out_dim, dropout, dropout_exit, dropout_p
        return VGG19MCEarlyExit(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type"))
    else:
        raise ValueError
    

class VGG(nn.Module):

    def __init__(self, blocks, num_class=100):
        super().__init__()
        self.blocks, self.non_sequentialized_blocks = blocks
        self.non_sequentialized_classifier = nn.ModuleList(modules = [nn.Linear(512, num_class)])
        self.classifier = nn.Sequential(*self.non_sequentialized_classifier)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block(output)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return [output]

def make_layers(cfg, batch_norm=False):
    blocks = nn.ModuleList()
    layers = nn.ModuleList()

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            blocks.append(layers)
            layers = nn.ModuleList()
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    non_sequentialized_blocks = blocks
    for block in range(len(blocks)):
        blocks[block] = nn.Sequential(*blocks[block])
    return (blocks, non_sequentialized_blocks)


class VGG19(VGG):
    def __init__(self, n_exits = 1, out_dim = 100, *args,  **kwargs):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        super().__init__(make_layers(cfg, batch_norm=True), num_class=out_dim, *args,  **kwargs)
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.init_weights()

class VGG19MC(VGG19):
    def __init__(self,dropout_exit = False, dropout = None, dropout_p = 0.5, n_exits = 1, out_dim = 100, *args,  **kwargs):
        super().__init__(n_exits = n_exits, out_dim = out_dim)
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.dropout_exit = dropout_exit

        if self.dropout is not None:
            self.blocks = nn.ModuleList()
            for block in range(len(self.non_sequentialized_blocks)):
                if self.dropout == "block":
                    dropout_layer = MCDropout(self.dropout_p)
                    self.non_sequentialized_blocks[block].append(dropout_layer)
                elif self.dropout == "layer":
                    new_block = []
                    # Can this be the same object?
                    dropout_layer = MCDropout(self.dropout_p)
                    for layer in self.non_sequentialized_blocks[block]:
                        new_block.append(layer)
                        new_block.append(dropout_layer)
                    self.non_sequentialized_blocks[block] = new_block
            # Overwrite blocks from super class
            for block in range(len(self.non_sequentialized_blocks)):
                self.blocks[block] = nn.Sequential(*self.non_sequentialized_blocks[block])
        if self.dropout_exit:
            self.classifier = nn.Sequential(MCDropout(self.dropout_p),self.classifier)
        
        self.init_weights()

    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block(output)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        self.intermediary_output_list = (output, [], 0, [])
        return [output]

class VGG19EarlyExit(VGG19):
    def __init__(self,n_exits = 4, out_dim = 100, *args,  **kwargs):
        super().__init__(n_exits = n_exits, out_dim = out_dim, *args,  **kwargs)
        ex1conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        ex1conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        ex1conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        ex1bn1 = nn.BatchNorm2d(128)
        ex1bn2 = nn.BatchNorm2d(256)
        ex1bn3 = nn.BatchNorm2d(512)
        ex1relu1 = nn.ReLU(inplace=True)
        ex1relu2 = nn.ReLU(inplace=True)
        ex1relu3 = nn.ReLU(inplace=True)
        ex1featureextractor_list = nn.ModuleList(modules = [ex1conv1,ex1bn1,ex1relu1,ex1conv2,ex1bn2,ex1relu2,ex1conv3,ex1bn3,ex1relu3])
        self.ex1featureextractor = nn.Sequential(*ex1featureextractor_list)
        self.ex1linear = nn.Linear(512, out_dim)

        ex2conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        ex2conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        ex2bn1 = nn.BatchNorm2d(256)
        ex2bn2 = nn.BatchNorm2d(512)
        ex2relu1 = nn.ReLU(inplace=True)
        ex2relu2 = nn.ReLU(inplace=True)
        ex2featureextractor_list = nn.ModuleList(modules = [ex2conv1,ex2bn1,ex2relu1,ex2conv2,ex2bn2,ex2relu2])
        self.ex2featureextractor = nn.Sequential(*ex2featureextractor_list)
        self.ex2linear = nn.Linear(512, out_dim)


        ex3conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        ex3bn1 = nn.BatchNorm2d(512)
        ex3relu1 = nn.ReLU(inplace=True)
        ex3featureextractor_list = nn.ModuleList(modules = [ex3conv1,ex3bn1,ex3relu1])
        self.ex3featureextractor = nn.Sequential(*ex3featureextractor_list)
        self.ex3linear = nn.Linear(512, out_dim)

        self.ex4linear = nn.Linear(512, out_dim)

        self.init_weights()

    def forward(self, x):
        out = self.blocks[0](x)

        out1 = self.ex1featureextractor(F.relu(out))
        out1 = F.avg_pool2d(out1, 2)
        middle1_fea = out1
        out1 = out1.view(out1.size(0), -1)
        out1 = self.ex1linear(out1)

        out = self.blocks[1](out)

        out2 = self.ex2featureextractor(F.relu(out))
        out2 = F.avg_pool2d(out2, 2)
        middle2_fea = out2
        out2 = out2.view(out2.size(0), -1)
        out2 = self.ex2linear(out2)

        out = self.blocks[2](out)

        out3 = self.ex3featureextractor(F.relu(out))
        out3 = F.avg_pool2d(out3, 2)
        middle3_fea = out3
        out3 = out3.view(out3.size(0), -1)
        out3 = self.ex3linear(out3)

        out = self.blocks[3](out)

        out4 = F.avg_pool2d(out, 2)
        middle4_fea = out4
        out4 = out4.view(out4.size(0), -1)
        out4 = self.ex4linear(out4)

        out = self.blocks[4](out)

        # ExitEnsemble used avg_pool2d here, but the og didn't...
        final_fea = out
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)

        self.intermediary_output_list = (out, [out1, out2, out3, out4], final_fea, [middle1_fea, middle2_fea, middle3_fea, middle4_fea])
        return [out1, out2, out3, out4, out]


class VGG19MCEarlyExit(VGG19EarlyExit):
    def __init__(self,dropout_exit = False, dropout = None, dropout_p = 0.5, n_exits = 4, out_dim = 100, *args,  **kwargs):
        super().__init__(n_exits = n_exits, out_dim = out_dim)
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.dropout_exit = dropout_exit

        if self.dropout is not None:
            for block in range(len(self.non_sequentialized_blocks)):
                if self.dropout == "block":
                    dropout_layer = MCDropout(self.dropout_p)
                    self.non_sequentialized_blocks[block].append(dropout_layer)
                elif self.dropout == "layer":
                    new_block = nn.ModuleList()
                    # Can this be the same object?
                    for layer in self.non_sequentialized_blocks[block]:
                        new_block.append(layer)
                        new_block.append(MCDropout(self.dropout_p))
                    self.non_sequentialized_blocks[block] = new_block
            # Overwrite blocks from super class
            for block in range(len(self.non_sequentialized_blocks)):
                self.blocks[block] = nn.Sequential(*self.non_sequentialized_blocks[block])

        if self.dropout_exit:
            self.ex1linear = nn.Sequential(MCDropout(self.dropout_p), self.ex1linear)
            self.ex2linear = nn.Sequential(MCDropout(self.dropout_p), self.ex2linear)
            self.ex3linear = nn.Sequential(MCDropout(self.dropout_p), self.ex3linear)
            self.ex4linear = nn.Sequential(MCDropout(self.dropout_p), self.ex4linear)
            # Overwrite classifier from super class
            self.classifier = nn.Sequential(MCDropout(self.dropout_p),self.classifier)
        
        self.init_weights()




        
        


                        




class MCDropout(nn.Dropout):

    def forward(self, x):
        return F.dropout(x, self.p, True, self.inplace)

