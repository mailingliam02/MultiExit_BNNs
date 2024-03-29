#from tests.test_early_exit import test_early_exit
# from inspect import isclass
# import models
# import torch
# import math
# import to_train
# from torch import nn
# from models.msdnet.msdnet_base import MsdNet
# from models.msdnet.msdnet_layers import MsdLayer, MsdLayer0
# from models.msdnet.msdnet_utils import MsdTransition, ConvBnRelu2d, View
# from tests.test_hyperparameters import get_msd_test_hyperparameters
# from tests.test_data import get_test_data
# from evaluate import evaluate

# def test_msdnet():
#     hyperparameters = get_msd_test_hyperparameters()
#     test_data, loader = get_test_data(hyperparameters)
#     model = models.get_network(hyperparameters["network"])
#     loss_fn = to_train.get_loss_function(hyperparameters["loss"])
#     # Not correct, but shouldn't be causing issues with training!
#     # test_architecture(model,hyperparameters["network"])
#     # Identical (Need to set weights of Linear layer)
#     #test_exit_outputs(model, test_data, hyperparameters["network"])
#     # Seems to be working, didn't actually work out by hand
#     #test_loss(model, test_data, loss_fn)
#     x = test_data[0]
#     y = test_data[1]
#     results = evaluate(loss_fn,loader,model,-1,0, 3)
#     print(results)
#     # actual = model(x)
#     # print("Actual")
#     # print(actual)
#     return



# def test_architecture(model, network_hyperparameters):
#     test_no_prune()
#     test_prune(model)
#     return None

# def test_exit_outputs(model, test_data, network_params):
#     x = test_data[0]
#     y = test_data[1]
#     actual = model(x)
#     print("Actual")
#     print(actual)
#     # Gets first exit output
#     expected = get_expected_exit_outputs(x)
#     print("Expected")
#     print(expected)
#     diff = torch.subtract(actual[0], expected)
#     # print("Blocks")
#     # print(model.blocks)
#     # print("Exits")
#     # print(model.exits)
#     # print("nplanes_tab")
#     # print(model._nplanes_tab)
#     print(diff)
#     return None

# def get_expected_exit_outputs(x):
#     start = ConvBnRelu2d(in_channels = 3,out_channels = 2,kernel_size = 3, padding = 1)
#     l1_s1_s2_str = ConvBnRelu2d(in_channels = 2,out_channels = 4,kernel_size = 3, stride = 2, padding = 1)
#     l1_l2_s2_reg = ConvBnRelu2d(in_channels = 4,out_channels = 2,kernel_size = 3, stride = 1, padding = 1)
#     l1_s1_l2_s2_str = ConvBnRelu2d(in_channels = 2,out_channels = 2,kernel_size = 3, stride = 2, padding = 1)
#     exit = Exit(8, out_dim = 10, inner_channels = 32)
#     modules = nn.ModuleList()
#     modules.append(start)
#     modules.append(l1_s1_s2_str)
#     modules.append(l1_s1_l2_s2_str)
#     modules.append(l1_l2_s2_reg)
#     modules.append(exit)
#     init_weights(modules)
#     x_l1_s1 = start(x)
#     x_l1_s2 = l1_s1_s2_str(x_l1_s1)
#     x_l2_s2_strided_part = l1_s1_l2_s2_str(x_l1_s1)
#     x_l2_s2_regular_part = l1_l2_s2_reg(x_l1_s2)
#     # Identity, regular part and strided part
#     out = [x_l1_s2, x_l2_s2_regular_part, x_l2_s2_strided_part]
#     return exit(torch.cat(out, dim=1))

# def Exit(n_channels, out_dim, inner_channels=None):
#     """Creates an intermediary exit"""
#     # Inner channels are the depth of the intermediary results
#     # of the two ConvBnRelu layers
#     inner_channels = inner_channels or n_channels
#     return nn.Sequential(
#         # Not explained why this structure of exit...
#         ConvBnRelu2d(n_channels, inner_channels, kernel_size=3,
#                         stride=2, padding=1),
#         ConvBnRelu2d(inner_channels, inner_channels, kernel_size=3,
#                         stride=2, padding=1),
#         nn.AvgPool2d(kernel_size=2),
#         # Flattens output
#         View(-1, inner_channels),
#         nn.Linear(inner_channels, out_dim),
#     )
# def init_weights(modules):
#     g_cpu = torch.Generator()
#     g_cpu.manual_seed(42)
#     print("Test Version")
#     for s in modules:
#         for m in s:
#             # Should check the inits are standard!
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0.0, math.sqrt(2/n), generator = g_cpu)
#                 m.bias.data.fill_(0.0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1.0)
#                 m.bias.data.fill_(0.0)
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.fill_(1.1)
#                 m.bias.data.fill_(0.0)
#             else:
#                 try:
#                     for j in m:
#                         if isinstance(j, nn.Conv2d):
#                             n = j.kernel_size[0] * j.kernel_size[1] * j.out_channels
#                             j.weight.data.normal_(0.0, math.sqrt(2/n), generator = g_cpu)
#                             j.bias.data.fill_(0.0)
#                         elif isinstance(j, nn.BatchNorm2d):
#                             j.weight.data.fill_(1.0)
#                             j.bias.data.fill_(0.0)
#                         elif isinstance(j, nn.Linear):
#                             j.weight.data.fill_(1.1)
#                             j.bias.data.fill_(0.0)
#                 except:
#                     pass



# def test_loss(model,test_data, loss_fn):
#     x = test_data[0]
#     y = test_data[1]
#     print(x.shape)
#     loss = loss_fn(model,x, y)
#     print(loss)
#     expected_loss = get_expected_loss(model, x, y)
#     return None

# def get_expected_loss(model,x,y):
    
#     return

# # def test_prune(model):
# #     hyperparams = dict(          # MSDNet architecture parameters
# #         call = 'MsdNet',
# #         in_shape = 32,
# #         out_dim = 10,
# #         n_scales = 3,
# #         n_exits = 3,
# #         nlayers_to_exit = 2,
# #         nlayers_between_exits = 2,
# #         nplanes_mulv = [2, 4, 8],
# #         nplanes_addh = 1,
# #         nplanes_init = 1,
# #         prune = "max",
# #         plane_reduction = 0.5, # Try this with 0 to avoid the halving
# #         exit_width = 32, # same as 128 dim 3x3 filters in exit?
# #         )
# #     model = models.get_network(hyperparams)
# #     # Will catch if prune is True, as adds a MsdTransition layer
# #     print(model._nplanes_tab)
# #     print(model._block_sep)
# #     print(model.blocks)
# #     assert len(model.blocks) == hyperparams["n_exits"]
# #     # Checks whether the correct number of scales have been inputted (invariant to changes in nlayer_to_exit or nlayers_between_exits)
# #     assert len(model.blocks[0][0].mods) == hyperparams["n_scales"]
# #     check_scales_removed(model, hyperparams["prune"])
# #     check_transition_layer_added(model)
# #     check_diagonal_block_created(model)
# #     if hyperparams["prune"] == "min":
# #         nlayers = len(model._nplanes_tab[0]) 

# #     # Checks the correct number of channels for first layer
# #     assert model.blocks[0][0].mods[1][0].in_channels == hyperparams["nplanes_mulv"][0]*hyperparams["nplanes_init"]
# #     assert model.blocks[0][0].mods[1][0].out_channels == hyperparams["nplanes_mulv"][1]*(hyperparams["nplanes_init"])
# #     return None

# # def check_scales_removed(model, prune_type):
# #     S = len(model.blocks)
# #     for i in range(S):
# #         # Get expected number of layers (forumla from Paper)
# #         # i starts from 0 in this loop rather than 1 (as in Paper), so no +1
# #         expected_n_scales = S-i 
# #         actual_n_scales = 0
# #         layer_module_list = get_layer_list(model.blocks[i][0])
# #         # Count the number of scales in a layer
# #         for j in range(len(layer_module_list)):
# #             print(layer_module_list[j])
# #             if layer_module_list[j] is not None:
# #                 actual_n_scales += 1            
# #         print(expected_n_scales, actual_n_scales)
# #         assert expected_n_scales == actual_n_scales
    

# #     #print(len(model.blocks[0][0]))
# #     assert model.blocks[0][0].mods[1]
# #     return None

# # def get_layer_list(layer):
# #     if isinstance(layer,MsdLayer) or isinstance(layer,MsdLayer0):
# #         layer_module_list = layer.mods
# #     elif isinstance(layer,MsdTransition):
# #         layer_module_list = layer.conv
# #     else:
# #         print(type(layer))
# #         raise TypeError
# #     return layer_module_list

# # def check_transition_layer_added(model):
# #     return None

# # def check_diagonal_block_created(model):
# #     return None


# # def test_no_prune():
# #     hyperparams = dict(          # MSDNet architecture parameters
# #         call = 'MsdNet',
# #         in_shape = 32,
# #         out_dim = 10,
# #         n_scales = 2,
# #         n_exits = 2,
# #         nlayers_to_exit = 2,
# #         nlayers_between_exits = 1,
# #         nplanes_mulv = [2, 4],
# #         nplanes_addh = 1,
# #         nplanes_init = 1,
# #         prune = "min",
# #         plane_reduction = 0, # Try this with 0 to avoid the halving
# #         exit_width = 32, # same as 128 dim 3x3 filters in exit?
# #         )
# #     model = models.get_network(hyperparams)
# #     print(model._nplanes_tab)
# #     print(model._block_sep)
# #     # Will catch if prune is True, as adds a MsdTransition layer
# #     assert len(model.blocks) == hyperparams["n_exits"]
# #     # Checks whether the correct number of scales have been inputted (invariant to changes in nlayer_to_exit or nlayers_between_exits)
# #     assert len(model.blocks[0][0].mods) == hyperparams["n_scales"]
# #     # Checks the correct number of channels for first layer
# #     assert model.blocks[0][0].mods[1][0].in_channels == hyperparams["nplanes_mulv"][0]*hyperparams["nplanes_init"]
# #     assert model.blocks[0][0].mods[1][0].out_channels == hyperparams["nplanes_mulv"][1]*(hyperparams["nplanes_init"])
# #     return None

# from model_tester import ResourceLoader
# from models.resnet18.resnet20 import ResNet20MCEarlyExit
# from to_train.loss import dict_drop, get_loss_function
# import torch
# import datasets
# resnet_20 = ResNet20MCEarlyExit()
# loader = ResourceLoader()
# loader_hyperparameters = dict(dataset_name = "cifar100",
# batch_size = (1,1,1), #(train, val, test) 
# # train and val batch sizes should be the same for plotting purposes
# augment = True,
# val_split = 0.1,
# )

# val_hyperparams = dict(       # train with classification loss only
#         call = 'ClassificationOnlyLoss',
#         n_exits = resnet_20.n_exits,
#         acc_tops = [1, 5],
#     )
# val_loss_fn = get_loss_function(val_hyperparams)
# train_loader, val_loader, test_loader = datasets.get_dataloader(loader_hyperparameters)
# print("Starting")
# def get_device(gpu):
#     return torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')

# def validate_model_acc(loss_f, net, val_iter, gpu):
#     metrics = []
#     with torch.no_grad():
#         for val_tuple in val_iter:
#             val_tuple = [t.to(get_device(gpu)) for t in val_tuple]
#             metrics += [loss_f.metrics(net, *val_tuple)]
#     return [sum(metric) / len(metric) for metric in zip(*metrics)]

# print(validate_model_acc(val_loss_fn, resnet_20,val_loader,-1))

# if __name__ == "__main__":
#     test_early_exit()
#     # test_msdnet()


import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):

    def __init__(self, blocks, num_class=100):
        super().__init__()
        self.blocks, self.non_sequentialized_blocks = blocks
        self.non_sequentialized_classifier = nn.ModuleList(modules = [nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_class)])
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
model = VGG19()
stuff = torch.load("tests/vgg19_bn-c79401a0.pth")
print(stuff.keys())
print(stuff['features.41.weight'])
key_list = ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']
for key in key_list:
    del stuff[key]

def vgg_pytorch_to_new_vgg(state_dict):
    block = 0
    layer = 0
    prev_feature_num = 0
    keys_list = list(state_dict.keys()).copy()
    for key in keys_list:
        key_split = key.split(".")
        if int(key_split[1]) - prev_feature_num == 3:
            block += 1
            layer = 0
        elif int(key_split[1]) - prev_feature_num > 0:
            layer += 1
        new_key = "blocks."+str(block)+"."+str(layer)+"."+key_split[-1]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
        prev_feature_num = int(key_split[1])
        if key_split[-1] == "running_var":
            layer += 1
    

#print(stuff)
#print(stuff.keys())


def make_layers(cfg, batch_norm = True):
    layers = nn.ModuleList()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
a = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
print(make_layers(a))
print(model)

print("model")
#model.load_state_dict(stuff, strict = False)


#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#print(model.state_dict().keys())

















# def dict_drop(dic, *keys):
#     new_dic = dic.copy()
#     for key in keys:
#         if key in new_dic:
#             del new_dic[key]
#     return new_dic

# def get_network(network_hyperparams):
#     if network_hyperparams["call"] == "MsdNet":
#         model = MsdNet(**dict_drop(network_hyperparams, "call"))
#     else:
#         raise AttributeError
#     return model

# if __name__ == "__main__":
#     hyperparams = dict(          # MSDNet architecture parameters
#             call = 'MsdNet',
#             in_shape = 32,
#             out_dim = 10,
#             n_scales = 3,
#             n_exits = 11,
#             nlayers_to_exit = 4,
#             nlayers_between_exits = 2,
#             nplanes_mulv = [6, 12, 24],
#             nplanes_addh = 1,
#             nplanes_init = 1,
#             prune = 'min',
#             plane_reduction = 0.5,
#             exit_width = 128,
#             btneck_widths = [4, 4, 4],
#             execute_exits = [0,1,2,3], # Determines which exits to return
#             )

#     model = get_network(hyperparams)
#     hyperparameters = dict(dataset_name = "cifar10",
#     batch_size = (1,1,1), #(train, val, test)
#     augment = True,
#     val_split = 0.2,
#     )
#     train_loader, val_loader, test_loader = datasets.get_dataloader(hyperparameters)
#     x,y = next(iter(val_loader))
#     output = model(x)
#     # print(len(output))
#     # print(output[0])
