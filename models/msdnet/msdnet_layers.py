import torch
from torch import nn
from models.msdnet.msdnet_utils import ConvBnRelu2d, MsdJoinConv

class MsdLayer0(nn.Module):
    def __init__(self, nplanes_list, in_shape):
        super().__init__()
        in_channels = 3
        self.mods = nn.ModuleList()
        
        if in_shape == 32:
            self.mods += [ConvBnRelu2d(in_channels, nplanes_list[0],
                                       kernel_size=3, padding=1)]
        elif in_shape == 224:
            conv = ConvBnRelu2d(in_channels, nplanes_list[0],
                                kernel_size=7, stride=2, padding=3)
            pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.mods += [nn.Sequential(conv, pool)]

        for i in range(1, len(nplanes_list)):
            self.mods += [ConvBnRelu2d(nplanes_list[i-1], nplanes_list[i],
                                       kernel_size=3, stride=2, padding=1)]
    def forward(self, x):
        out = [x]
        for i in range(len(self.mods)):
            out += [self.mods[i](out[i])]
        return out[1:]

    
class MsdLayer(nn.Module):
    def __init__(self, nplanes_tab, btneck_widths):
        super().__init__()
        in_scales, out_scales = nplanes_tab.astype(bool).sum(axis=0)
        assert in_scales - out_scales <= 1
        
        if not btneck_widths:
            btneck_widths = [None] * len(nplanes_tab)
    
        self.mods = nn.ModuleList()
        for i, (n_in, n_out) in enumerate(nplanes_tab):
            n_in_prev = nplanes_tab[i-1, 0] if i else 0
            btneck_width_prev = btneck_widths[i-1] if i else None
            self.mods += [MsdJoinConv(n_in, n_in_prev, n_out - n_in,
                                      btneck_widths[i], btneck_width_prev)
                          if n_out else None]
            
    def forward(self, x):
        out = []
        for i, m in enumerate(self.mods):
            x_down = None if i == 0 else x[i-1]
            out += [m(x[i], x[i], x_down) if m else None]
        return out