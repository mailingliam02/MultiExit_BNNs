import torch
from torch.nn import functional as F
from to_train.loss.base_classes import _MultiExitAccuracy


class ClassificationOnlyLoss(_MultiExitAccuracy):
    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = net(X)
        self._cache['y'] = y
        return sum(F.cross_entropy(logits, y)
                   for logits in self._cache['logits_list'])

    
class DistillationBasedLoss(_MultiExitAccuracy):    
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.05,
                 global_scale=1.0):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0
        self.global_scale = global_scale
        
        self.metric_names += ['adj_maxprob', 'temperature'] 

    def __call__(self, net, X, y, *args):
        logits_list = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        cum_loss = (1.0 - self.C) * F.cross_entropy(logits_list[-1], y)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        for logits in logits_list[:-1]:
            logprob_s = F.log_softmax(logits / self.T, dim=1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss

        self._cache['adj_maxprob'] = adj_maxprob = prob_t.max(dim=1)[0].mean()
        if adj_maxprob > self.maxprob:
            self.T *= self.Tmult
        return cum_loss * self.global_scale

    def metrics(self, net, X, y, *args):
        logits_list = net.train(False)(X)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        adj_maxprob = prob_t.max(dim=1)[0].mean()
        return self._metrics(logits_list, y) + [adj_maxprob, self.T]

    def trn_metrics(self):
        out = self._metrics(self._cache['logits_list'], self._cache['y'])
        return out + [self._cache['adj_maxprob'], self.T]


class DistillationLossConstTemp(_MultiExitAccuracy):
    def __init__(self, C, T, n_exits, acc_tops=(1,), global_scale=1.0):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.T = T
        self.global_scale = global_scale
        self.metric_names += ['adj_maxprob']

    def __call__(self, net, X, y, *args):
        logits_list = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        cum_loss = (1.0 - self.C) * F.cross_entropy(logits_list[-1], y)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        self._cache['prob_t'] = prob_t
        for logits in logits_list[:-1]:
            logprob_s = F.log_softmax(logits / self.T, dim=1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss
        return cum_loss * self.global_scale

    def metrics(self, net, X, y, *args):
        logits_list = net.train(False)(X)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        adj_maxprob = prob_t.max(dim=1)[0].mean()
        return self._metrics(logits_list, y) + [adj_maxprob]

    def trn_metrics(self):
        out = self._metrics(self._cache['logits_list'], self._cache['y'])
        adj_maxprob = self._cache['prob_t'].max(dim=1)[0].mean()
        return out + [adj_maxprob]

class MultiExitAccuracy(_MultiExitAccuracy): 
    def __init__(self, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)
