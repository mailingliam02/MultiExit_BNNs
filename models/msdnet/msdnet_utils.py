import collections
import torch
from torch import nn

def ConvBnRelu2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    in_channels = int(in_channels)
    out_channels = int(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(*self.shape)

class MsdTransition(nn.Module):
    def __init__(self, nplanes_tab):
        super().__init__()
        self.conv = nn.ModuleList([ConvBnRelu2d(n_in, n_out, kernel_size=1)
                                   if n_in else None
                                   for n_in, n_out in nplanes_tab])
    def forward(self, inputs):
        return [m(x) if m else None for m, x in zip(self.conv, inputs)]

class MsdJoinConv(nn.Module):
    def __init__(self, n_in, n_in_down, n_out, btneck, btneck_down):
        super().__init__()
        # Creates the strided convolution operation
        if n_in_down:
            assert n_out % 2 == 0
            n_out //= 2
            self.conv_down = self._btneck(n_in_down, n_out, stride=2,
                                          btneck=btneck_down)
        # Creates the standard convolution operation
        self.conv = self._btneck(n_in, n_out, stride=1, btneck=btneck)

    def _btneck(self, n_in, n_out, stride, btneck:int):
        # Bottle neck which specifies whether to have two Convs or just 1
        # Not sure where this is discussed in papers
        if btneck:
            n_mid = min(n_in, btneck * n_out)
            return nn.Sequential(
                ConvBnRelu2d(n_in, n_mid, 1),
                ConvBnRelu2d(n_mid, n_out, 3, stride=stride, padding=1))
        else:
            return ConvBnRelu2d(n_in, n_out, 3, stride=stride, padding=1)

    def forward(self, x1, x2, x_down):
        # Identity x^s_l-1 and regular convolution h(x^s_l)
        out = [x1, self.conv(x2)]
        # Strided convolution h~(x^s-1_l)
        out += [self.conv_down(x_down)] if x_down is not None else []
        return torch.cat(out, dim=1)


class TransientDict:
    def __init__(self, _keep=(), **kw):
        self._dic = collections.OrderedDict(**kw)
        self._keep = _keep

    def __iter__(self):
        return self._dic.__iter__()

    def __repr__(self):
        items_str = '\n'.join(str(i) for i in self._dic.items())
        return 'TransientDict([\n{}\n])'.format(items_str)

    def __delitem__(self, key):
        del self._dic[key]

    def __setitem__(self, key, val):
        self._dic[key] = val

    def __getitem__(self, key):
        if key == -1:
            k, v = self._dic.popitem()
            self._dic[k] = v if k in self._keep else None
            return v
        else:
            val = self._dic[key]
            if key not in self._keep: self._dic[key] = None
            return val

    def keys(self):
        return self._dic.keys()


# From https://github.com/pytorch/pytorch/issues/19808#
class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs