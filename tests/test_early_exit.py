from model_tester import ResourceLoader, OverthinkingEvaluation
import torch
import numpy as np
def test_early_exit():
    np.random.seed(45)
    model = Model()
    test_loader = iter([(torch.tensor([[1],[1]]),torch.tensor([[0,0]])),(torch.tensor([[1],[1]]),torch.tensor([[0,0]])),
        (torch.tensor([[1],[1]]),torch.tensor([[0,0]])),(torch.tensor([[1],[1]]),torch.tensor([[0,0]]))])
    oe = OverthinkingEvaluation(model, test_loader, gpu=-1, mc_dropout = False, mc_passes = 2, ensemble = True)
    print(oe.layer_correct)
    print(oe.layer_wrong)
    print(oe.layer_predictions)
    print(oe.layer_confidence)
    oe.theoretical_best_performance()
    return None

class Model():
    def __init__(self):
        self.n_exits = 2
        self.out_dim = 2
    def eval(self):
        pass
    def __call__(self,x):
        output = [torch.from_numpy(np.random.rand(x.shape[0],2)),torch.from_numpy(np.random.rand(x.shape[0],2))]
        print(output)
        return output