from xmlrpc.client import FastUnmarshaller
from torch import nn
from typing import List
import numpy as np
import math
import torch
from models.msdnet.msdnet_utils import ConvBnRelu2d, View, MsdTransition, MsdJoinConv, TransientDict
from models.msdnet.msdnet_layers import MsdLayer, MsdLayer0
from models.mcdropout import get_dropout

class _TraceInForward(nn.Module):
    def forward(self, x, keep_layers=()):
        raise NotImplementedError
    
    def trace(self, *inputs, keep_layers, **kw_inputs):
        self.forward(*inputs, keep_layers, **kw_inputs)
        return self._trace

class MsdNet(_TraceInForward):
    def __init__(self, in_shape, out_dim, n_scales, n_exits, nlayers_to_exit,
                 nlayers_between_exits, nplanes_mulv:List[int],
                 nplanes_addh:int, nplanes_init=32, prune=None,
                 plane_reduction=0.0, exit_width=None, btneck_widths=(),
                 execute_exits=None, test_mode = False, dropout = None, 
                 dropout_p = 0.5, dropout_exit = False):
        """Creates the Multi-Scale DenseNet
        Attributes
        ----------
        in_shape : np arr
            x
        out_dim : str
            x
        n_scales : int
            x
        n_exits : str
            x
        n_layers_to_exit : int
            x
        nlayers_between_exits : str
            x
        nplanes_mulv : list of int
            Specifies the multiples of the number of channels in each scale
        nplanes_addh : init
            By how many multiples of nplanes_mulv each subsequent scale 
            increases by
        nplanes_init : int
            The starting number of channels as the first layer multiplier of
            the nplanes_mulv values
        prune : str
            x
        plane_reduction : int
            x
        exit_width : int
            x
        btneck_widths : int
            x
        execute_exits : int
            x    
        dropout : str
            scale, layer, block, None
        dropout_p : float
            x
        dropout_exit : bool
            Bool to include dropout on classifier
        """
        # Init for trace method
        super().__init__()
        # Why is this assertion necessary?
        assert nlayers_to_exit >= nlayers_between_exits
        # Test mode
        self.test_mode = test_mode
        # Define number of exit classifiers
        self.n_exits = n_exits
        # Define size of output of each classifier
        self.out_dim = out_dim
        # Define size of result during Conv before FC in exits
        self.exit_width = exit_width
        # Defines dropout probability
        self.dropout = dropout
        self.dropout_exit = dropout_exit
        self.p = dropout_p
        # Specifies which exits to exit at
        self._execute_exits = (execute_exits if execute_exits is not None
                               else range(n_exits))
        # Caculates list of layers per block
        block_nlayers = [nlayers_to_exit] + [nlayers_between_exits]*(n_exits-1)
        # Number of layers
        n_layers = 1 + sum(block_nlayers)
        print("nlayers", n_layers)
        # Extracts an array of each scale and layer's channels
        nplanes_tab = self.nplanes_tab(n_scales, n_layers, nplanes_init,
                                       nplanes_mulv, nplanes_addh, prune,
                                       plane_reduction)
        self._nplanes_tab = nplanes_tab
        # Get list of which layers belong in which block
        self._block_sep = block_sep = self.block_sep(block_nlayers, n_scales,
                                                    prune, plane_reduction)     
        self.blocks = nn.ModuleList()
        self.exits = nn.ModuleList()
        # Creates the Blocks and Exits
        for i in range(n_exits):
            # Gets all the channels for each scale, across each layer in block
            self.blocks.append(self.Block(
                nplanes_tab[:, block_sep[i]-1:block_sep[i+1]],
                in_shape if i == 0 else 0, btneck_widths))
            self.exits.append(self.Exit(nplanes_tab[-1,block_sep[i+1]-1],
                                      out_dim, exit_width))
        self.init_weights()

    def Block(self, nplanes_tab, layer0_size, btneck_widths):
        """Creates a Block as illustrated in Figure 9 of MsdNet paper."""
        block = []
        # Adds block if input size is specified
        if layer0_size:
            block = [MsdLayer0(nplanes_tab[:,0], layer0_size, dropout = self.dropout, dropout_p = self.p)]
        # Adds additional layers and if scale is removed, add transition layer
        for i in range(1, nplanes_tab.shape[1]):
            # Adds all scales in a column simultaneously
            block += [MsdLayer(nplanes_tab[:,i-1:i+1], btneck_widths, dropout = self.dropout, dropout_p = self.p)
                      if nplanes_tab[-1,i-1] < nplanes_tab[-1,i] else
                      MsdTransition(nplanes_tab[:,i-1:i+1])]
            if self.dropout == "layer":
                # Gets added for each layer in block
                block += [get_dropout(p = self.p)]
        if self.dropout == "block":
            # Gets added only once at end of block
            block += [get_dropout(p = self.p)]
        return nn.Sequential(*block)

    def Exit(self, n_channels, out_dim, inner_channels=None):
        """Creates an intermediary exit"""
        # Inner channels are the depth of the intermediary results
        # of the two ConvBnRelu layers
        inner_channels = inner_channels or n_channels
        if self.dropout_exit:
            return nn.Sequential(
                # Not explained why this structure of exit...
                ConvBnRelu2d(n_channels, inner_channels, kernel_size=3,
                            stride=2, padding=1),
                ConvBnRelu2d(inner_channels, inner_channels, kernel_size=3,
                            stride=2, padding=1),
                nn.AvgPool2d(kernel_size=2),
                # Flattens output
                View(-1, inner_channels),
                get_dropout(p = self.p),
                nn.Linear(inner_channels, out_dim),
            )
        else:
            return nn.Sequential(
                # Not explained why this structure of exit...
                ConvBnRelu2d(n_channels, inner_channels, kernel_size=3,
                            stride=2, padding=1),
                ConvBnRelu2d(inner_channels, inner_channels, kernel_size=3,
                            stride=2, padding=1),
                nn.AvgPool2d(kernel_size=2),
                # Flattens output
                View(-1, inner_channels),
                nn.Linear(inner_channels, out_dim),
            )

    def block_sep(self, block_nlayers, n_scales, prune, plane_reduction):
        """Identifies the indices of layers for each block"""
        n_layers = 1 + sum(block_nlayers)
        # Identify the layers which are reduced in scale
        reduce_layers = self._reduce_layers(n_scales, n_layers, prune,
                                           plane_reduction)
        # Creates the layer indices belonging to each block
        sep = np.cumsum([1] + block_nlayers)
        # Add an extra transition layer for that block
        # if scale gets reduced in the block
        shift = np.zeros_like(sep)
        # Shifts the subsequent indices
        for i in reduce_layers:
            shift += (sep >= i)
        return sep + shift

    def nplanes_tab(self, n_scales, n_layers, nplanes_init, nplanes_mulv,
                    nplanes_addh, prune, plane_reduction):
        """Tracks number of channels at each layer and scale"""
        # Identifies which layers to reduce scale
        reduce_layers = self._reduce_layers(n_scales, n_layers, prune,
                                           plane_reduction)
        # Identifies which scales are not needed (in final layer)
        nprune_per_layer = self._nprune_per_layer(n_scales, n_layers, prune)
        # Creates the large array containing every channel at each layer (column) and scale (row)
        hbase, nprune = [nplanes_init], [0]
        for i in range(1, n_layers):
            hbase += [hbase[-1] + nplanes_addh]
            nprune += [nprune_per_layer[i]]
            if i in reduce_layers:
                # Not sure this behavior is intended, especially if the scale isn't removed!
                # (Currently with prune = 'min' scales do not get removed at the intervals,
                # just have the channels reduced by half)
                hbase += [math.floor(hbase[-1] * plane_reduction)]
                nprune += [nprune_per_layer[i]]
        # Multiplies each scales (row) by the number of channels after input
        planes_tab = np.outer(nplanes_mulv, hbase)
        # Prunes the final scales which are unneeded
        for i in range(len(hbase)):
            planes_tab[:nprune[i], i] = 0
        return planes_tab

    def _reduce_layers(self, n_scales, n_layers, prune, plane_reduction):
        if not plane_reduction:
            return []
        # Only reduces scales twice at 1/3 and 2/3 interval
        elif prune == 'min':
            return [math.floor((n_layers-1)*1/3),
                    math.floor((n_layers-1)*2/3)]
        # Reduces scales until only one scale is left
        elif prune == 'max':
            interval = math.ceil((n_layers-1) / n_scales)
            return list(range(interval+1, n_layers, interval))

    def _nprune_per_layer(self, n_scales, n_layers, prune):
        # Only prunes the unused scales in last block
        if prune == 'min':
            nprune = min(n_scales, n_layers) - np.arange(n_layers, 0, -1)
            return list(np.maximum(0, nprune))
        # Actually removes the scales at each interval
        elif prune == 'max':
            interval = math.ceil((n_layers-1) / n_scales)
            return [0] + [math.floor(i/interval) for i in range(n_layers-1)]
        else:
            return [0] * n_layers
        
    def init_weights(self):
        if self.test_mode:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(42)
        for m in self.modules():
            # Should check the inits are standard!
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if self.test_mode:
                    m.weight.data.normal_(0.0, math.sqrt(2/n), generator = g_cpu)
                else:
                    m.weight.data.normal_(0.0, math.sqrt(2/n))
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                if self.test_mode:
                    n = m.in_features*m.out_features
                    m.weight.data.normal_(0.0, math.sqrt(2/n), generator = g_cpu)
                m.bias.data.fill_(0.0)

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