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
        return nn.functional.dropout(x, p = self.dropout_p, training = True,
            inplace = self.inplace_bool)

if __name__ == "__main__":
    x = get_dropout()
    