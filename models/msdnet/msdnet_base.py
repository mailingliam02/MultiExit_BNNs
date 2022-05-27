import torch
from torch import nn
from typing import List
import numpy as np
import math
from models.msdnet.msdnet_utils import ConvBnRelu2d, View, MsdTransition, MsdJoinConv, TransientDict
from models.msdnet.msdnet_layers import MsdLayer, MsdLayer0

class _TraceInForward(nn.Module):
    def forward(self, x, keep_layers=()):
        raise NotImplementedError
        self._trace = TransientDict()
    
    def trace(self, *inputs, keep_layers, **kw_inputs):
        self.forward(*inputs, keep_layers, **kw_inputs)
        return self._trace

class MsdNet(_TraceInForward):
    def __init__(self, in_shape, out_dim, n_scales, n_exits, nlayers_to_exit,
                 nlayers_between_exits, nplanes_mulv:List[int],
                 nplanes_addh:int, nplanes_init=32, prune=None,
                 plane_reduction=0.0, exit_width=None, btneck_widths=(),
                 execute_exits=None):

        super().__init__()
        assert nlayers_to_exit >= nlayers_between_exits
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.exit_width = exit_width
        self._execute_exits = (execute_exits if execute_exits is not None
                               else range(n_exits))
        
        block_nlayers = [nlayers_to_exit] + [nlayers_between_exits]*(n_exits-1)
        n_layers = 1 + sum(block_nlayers)
        nplanes_tab = self.nplanes_tab(n_scales, n_layers, nplanes_init,
                                       nplanes_mulv, nplanes_addh, prune,
                                       plane_reduction)
        self._nplanes_tab = nplanes_tab
        self._block_sep = block_sep = self.block_sep(block_nlayers, n_scales,
                                                    prune, plane_reduction)
        self.blocks = nn.ModuleList()
        self.exits = nn.ModuleList()
        for i in range(n_exits):
            self.blocks.append(self.Block(
                nplanes_tab[:, block_sep[i]-1:block_sep[i+1]],
                in_shape if i == 0 else 0, btneck_widths))
            self.exits.append(self.Exit(nplanes_tab[-1,block_sep[i+1]-1],
                                      out_dim, exit_width))
        self.init_weights()

    def Block(self, nplanes_tab, layer0_size, btneck_widths):
        block = []
        if layer0_size:
            block = [MsdLayer0(nplanes_tab[:,0], layer0_size)]
        for i in range(1, nplanes_tab.shape[1]):
            block += [MsdLayer(nplanes_tab[:,i-1:i+1], btneck_widths)
                      if nplanes_tab[-1,i-1] < nplanes_tab[-1,i] else
                      MsdTransition(nplanes_tab[:,i-1:i+1])]
        return nn.Sequential(*block)

    def Exit(self, n_channels, out_dim, inner_channels=None):
        inner_channels = inner_channels or n_channels
        return nn.Sequential(
            ConvBnRelu2d(n_channels, inner_channels, kernel_size=3,
                         stride=2, padding=1),
            ConvBnRelu2d(inner_channels, inner_channels, kernel_size=3,
                         stride=2, padding=1),
            nn.AvgPool2d(kernel_size=2),
            View(-1, inner_channels),
            nn.Linear(inner_channels, out_dim),
        )

    def block_sep(self, block_nlayers, n_scales, prune, plane_reduction):
        n_layers = 1 + sum(block_nlayers)
        reduce_layers = self._reduce_layers(n_scales, n_layers, prune,
                                           plane_reduction)
        sep = np.cumsum([1] + block_nlayers)
        shift = np.zeros_like(sep)
        for i in reduce_layers:
            shift += (sep >= i)
        return sep + shift

    def nplanes_tab(self, n_scales, n_layers, nplanes_init, nplanes_mulv,
                    nplanes_addh, prune, plane_reduction):
        
        reduce_layers = self._reduce_layers(n_scales, n_layers, prune,
                                           plane_reduction)
        nprune_per_layer = self._nprune_per_layer(n_scales, n_layers, prune)
        hbase, nprune = [nplanes_init], [0]
        for i in range(1, n_layers):
            hbase += [hbase[-1] + nplanes_addh]
            nprune += [nprune_per_layer[i]]
            if i in reduce_layers:
                hbase += [math.floor(hbase[-1] * plane_reduction)]
                nprune += [nprune_per_layer[i]]
                
        planes_tab = np.outer(nplanes_mulv, hbase)
        for i in range(len(hbase)):
            planes_tab[:nprune[i], i] = 0
        return planes_tab

    def _reduce_layers(self, n_scales, n_layers, prune, plane_reduction):
        if not plane_reduction:
            return []
        elif prune == 'min':
            return [math.floor((n_layers-1)*1/3),
                    math.floor((n_layers-1)*2/3)]
        elif prune == 'max':
            interval = math.ceil((n_layers-1) / n_scales)
            return list(range(interval+1, n_layers, interval))

    def _nprune_per_layer(self, n_scales, n_layers, prune):
        if prune == 'min':
            nprune = min(n_scales, n_layers) - np.arange(n_layers, 0, -1)
            return list(np.maximum(0, nprune))
        elif prune == 'max':
            interval = math.ceil((n_layers-1) / n_scales)
            return [0] + [math.floor(i/interval) for i in range(n_layers-1)]
        else:
            return [0] * n_layers
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0.0, math.sqrt(2/n))
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.bias.data.fill_(0.0)

    def forward(self, x, keep_layers=()):
        max_block = max(self._execute_exits)
        logits = []
        x = TransientDict(x=x, _keep=keep_layers)
        for i in range(max_block+1):
            h = self.blocks[i](x[-1])
            x[f'block{i}'] = h
            if i in self._execute_exits:
                logits += [self.exits[i](h[-1])]
            else:
                logits += [()]
        x[-1]
        x['logits'] = logits
        self._trace = x
        return x['logits']
