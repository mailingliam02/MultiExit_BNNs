import torch
from torch import nn

def get_dropout(p = 0.5, inplace = False, dropout_type = "standard", mc = True):
    # Returns a nn.Module
    if dropout_type == "standard" and mc == True:
        return MCDropout(p = p, inplace = inplace)



class MCDropout(nn.Dropout):
    def __init__(self, p = 0.5, inplace = False):
        super().__init__(p = p, inplace = inplace)
        self.dropout_p = p
        self.inplace_bool = inplace
    
    def forward(self,x):
        print(type(x))
        # If Dropout is applied in block, will be given outputs for each of the scales in a list!
        if type(x) == list:
            output = []
            for input in x:
                if input is not None:
                    output.append(nn.functional.dropout(input, p = self.dropout_p, training = True,
                        inplace = self.inplace_bool))
                else:
                    output.append(None)
        else:
            output = nn.functional.dropout(x, p = self.dropout_p, training = True,
                inplace = self.inplace_bool)
        return output
        
def test_dropout():
    dropout = get_dropout(p = 0.2)
    dropout.train()
    x = torch.rand((1,1,1,10))
    out = dropout(x)
    dropoutv2 = nn.Dropout(0.2,False)
    dropoutv2.train()
    outv2 = dropoutv2(x)
    print(out,outv2)
    dropout.eval()
    dropoutv2.eval()
    out = dropout(x)
    outv2 = dropoutv2(x)
    # Still works in eval mode!
    print(out,outv2)




    