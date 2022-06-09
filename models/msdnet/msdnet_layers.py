import torch
from torch import nn
from models.msdnet.msdnet_utils import ConvBnRelu2d, MsdJoinConv

class MsdLayer0(nn.Module):
    """Creates the first layer of MsdNet
    Attributes
    ----------
    nplanes_list : np arr
        
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """
    def __init__(self, nplanes_list, in_shape, dropout = None, dropout_p = 0.5):
        super().__init__()
        in_channels = 3
        self.mods = nn.ModuleList()
        
        # For Cifar
        if in_shape == 32:
            module = ConvBnRelu2d(in_channels, nplanes_list[0],
                                       kernel_size=3, padding=1)
        
        # For ImageNet
        elif in_shape == 224:
            conv = ConvBnRelu2d(in_channels, nplanes_list[0],
                                kernel_size=7, stride=2, padding=3)
            pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            module = nn.Sequential(conv, pool)

        # Add dropout after every Convolution
        if dropout == "scale":
            dropout_layer = nn.Dropout(p = dropout_p)
            module = nn.Sequential(module, dropout_layer)

        self.mods += [module]
        # Strided Convolutions connecting scales within a layer
        for i in range(1, len(nplanes_list)):
            conv = ConvBnRelu2d(nplanes_list[i-1], nplanes_list[i],
                                       kernel_size=3, stride=2, padding=1)
            if dropout == "scale":
                dropout_layer = nn.Dropout(p = dropout_p)
                self.mods += [nn.Sequential(conv, dropout_layer)]
            else:
                self.mods += [conv]

    def forward(self, x):
        out = [x]
        for i in range(len(self.mods)):
            # Strided convolution applied consecutively to each output
            out += [self.mods[i](out[i])]
        # Returns all but first output
        return out[1:]

    
class MsdLayer(nn.Module):
    def __init__(self, nplanes_tab, btneck_widths, dropout = None, dropout_p = 0.5):
        """Creates the Multi-Scale DenseNet
        Attributes
        ----------
        nplanes_tab : np arr
            A n_in_scale x n_out_scale arr of channels at each scale and layer
        btneck_widths : 
            family name of the person 
        """
        super().__init__()
        # Gets number of scales at input vs output
        # if channel in nplanes_tab is 0, scale is not counted
        in_scales, out_scales = nplanes_tab.astype(bool).sum(axis=0)
        # Make sure more in_scales than out_scales
        assert in_scales - out_scales <= 1
        # Initialize bottle necks if not specified
        if not btneck_widths:
            btneck_widths = [None] * len(nplanes_tab)
        self.mods = nn.ModuleList()
        # Joins the identity, the regular conv and the strided conv performed 
        # in each layer
        for i, (n_in, n_out) in enumerate(nplanes_tab):
            n_in_prev = nplanes_tab[i-1, 0] if i else 0
            # Gets bottle neck for the strided conv
            btneck_width_prev = btneck_widths[i-1] if i else None
            if n_out:
                conv = MsdJoinConv(n_in, n_in_prev, n_out - n_in,
                                        btneck_widths[i], btneck_width_prev)
                if dropout == "scale":
                    # Adds dropout for each scale in layer
                    dropout_layer = nn.Dropout(p = dropout_p)
                    module = [nn.Sequential(conv, dropout_layer)]
                else:
                    module = [conv]
            else:
                module = None
            self.mods += module
            
    def forward(self, x):
        out = []
        for i, m in enumerate(self.mods):
            # Does not downsample for the first scale
            x_down = None if i == 0 else x[i-1]
            out += [m(x[i], x[i], x_down) if m else None]
        return out