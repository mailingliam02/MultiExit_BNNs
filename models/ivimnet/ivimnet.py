import torch
import math
from torch import nn
from models.msdnet.msdnet_utils import TransientDict

class IVIMNet(nn.Module):
    def __init__(self, input_size, output_size, n_exits, n_layers, neurons_per_layer, 
                layers_with_exits, test_mode = False, dropout = None, 
                 dropout_p = 0.5, dropout_exit = False):
        # Init for module assigning
        super().__init__()
        self.layers_list = self.define_structure(n_exits, n_layers, neurons_per_layer, layers_with_exits)
        self._execute_exits = layers_with_exits
        self.neurons_per_layer = neurons_per_layer
        self.input_size = input_size
        self.output_size = output_size
        self.test_mode = test_mode
        self.blocks = nn.ModuleList()
        self.exits = nn.ModuleList()
        print(self.layers_list)
        for i in range(n_exits):
            # Gets all the channels for each scale, across each layer in block
            self.blocks.append(self.Block(self.layers_list[i], True if i == 0 else False))
            self.exits.append(self.Exit(self.layers_list[i][-1]))
        self.init_weights()
        return

    def define_structure(self, n_exits, n_layers, neurons_per_layer, layers_with_exits):
        layers = range(n_layers)
        layers_list = [[] for _ in range(len(layers_with_exits))]
        block = 0
        for layer in layers:
            # What to end current block on
            layers_list[block].append(layer)
            if layer in layers_with_exits:
                block += 1
        return layers_list

    def Block(self, layers, first_block):
        mods = []
        if first_block:
            lin = nn.Linear(self.input_size, self.neurons_per_layer[0], bias = True)
            batch_norm = nn.BatchNorm1d(self.neurons_per_layer[0])
            elu = nn.ELU()
            mods += [nn.Sequential(lin,batch_norm,elu)]
        for layer in layers:
            lin = nn.Linear(self.neurons_per_layer[layer-1],self.neurons_per_layer[layer], bias = True)
            batch_norm = nn.BatchNorm1d(self.neurons_per_layer[layer])
            elu = nn.ELU()
            mods += [nn.Sequential(lin,batch_norm,elu)]
        return nn.Sequential(*mods)
    
    def Exit(self, exit_input_size):
        return nn.Sequential(nn.Linear(exit_input_size, self.output_size))
    
    def init_weights(self):
        if self.test_mode:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(42)
        for m in self.modules():
            # Should check the inits are standard!
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                if self.test_mode:
                    n = m.in_features*m.out_features
                    m.weight.data.normal_(0.0, math.sqrt(2/n), generator = g_cpu)
                m.bias.data.fill_(0.0)
        return

    def forward(self, x, keep_layers=()):
        max_block = max(self._execute_exits)
        logits = []
        x = TransientDict(x=x, _keep=keep_layers)
        for i in range(max_block+1):
            # Results from each block 
            # list with each element being output for each scale
            h = self.blocks[i](x[-1])
            # Stores latest results
            x[f'block{i}'] = h
            # Exits with coarsest feature map
            if i in self._execute_exits:
                logits += [self.exits[i](h[-1])]  # At this point could check threshold to exit
            else:
                logits += [()]
        # Deletes intermediary results
        x[-1]
        # Stores exit outputs
        x['logits'] = logits
        # Saves dictionary for trace method from super
        self._trace = x
        return x['logits']