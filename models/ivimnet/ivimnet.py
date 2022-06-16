import torch
from torch import nn
from models.msdnet.msdnet_utils import TransientDict

class IVIMNet():
    def __init__(self, n_exits, n_layers, neurons_per_layer, layers_with_exits):
        self.layers_list = define_structure(n_exits, n_exits, n_layers, neurons_per_layer, layers_with_exits)
        self._execute_exits = layers_with_exits
        self.neurons_per_layer = neurons_per_layer
        self.blocks = nn.ModuleList()
        self.exits = nn.ModuleList()

        for i in range(n_exits):
            # Gets all the channels for each scale, across each layer in block
            self.blocks.append(self.Block(self.layers_list[i], True if i == 0 else False))
            self.exits.append(self.Exit())
        self.init_weights()
        return

    def define_structure(self, n_exits, n_layers, neurons_per_layer, layers_with_exits):
        layers = range(n_layers)
        layers_list = [[] for _ in range(len(layers_with_exits))]
        block = 0
        for layer in layers:
            if layer in layers_with_exits:
                block += 1
            layers_list[block].append(layer)
        return layers_list

    def Block(self, layers, first_block):
        mods = []
        if first_block:
            mods += [layer_0]
        for layer in layers:
            mods += [nn.Linear(self.neurons_per_layer[layer-1],self.neurons_per_layer[layer], bias = True)]
        return nn.Sequential(*block)
    
    def Exit(self):
        return
    
    def init_weights():
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