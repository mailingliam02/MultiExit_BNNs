import torch
from to_train.loss.base_classes import _MultiExitAccuracy
# https://github.com/hjdw2/Exit-Ensemble-Distillation/blob/main/train.py
class ExitEnsembleDistillation(_MultiExitAccuracy):
    def __init__(self, n_exits, acc_tops=(1,), use_EED = True, loss_output = "MSE", use_feature_dist = False, temperature = 3):
        super().__init__(n_exits, acc_tops)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.use_EED = use_EED
        self.loss_output = loss_output
        self.use_feature_dist = use_feature_dist
        self.temperature = temperature


    def __call__(self, net, X, y):
        _ = net(X)
        output, middle_output1, middle_output2, middle_output3, \
            final_fea, middle1_fea, middle2_fea, middle3_fea = net.intermediary_output_list
        target = y
        loss = self.criterion(output, target)
        middle1_loss = self.criterion(middle_output1, target)
        middle2_loss = self.criterion(middle_output2, target)
        middle3_loss = self.criterion(middle_output3, target)
        L_C = loss + middle1_loss + middle2_loss + middle3_loss

        if self.use_EED:
            target_output = (middle_output1/4 + middle_output2/4 + middle_output3/4 + output/4).detach()
            target_fea = (middle1_fea/4 + middle2_fea/4 + middle3_fea/4 + final_fea/4).detach()
        else:
            target_output = output.detach()
            target_fea = final_fea.detach()

        if self.loss_output == 'KL':
            temp = target_output / self.temperature
            temp = torch.softmax(temp, dim=1)
            loss1by4 = self.kd_loss_function(middle_output1, temp) * (self.temperature**2)
            loss2by4 = self.kd_loss_function(middle_output2, temp) * (self.temperature**2)
            loss3by4 = self.kd_loss_function(middle_output3, temp) * (self.temperature**2)
            L_O = 0.1 * (loss1by4 + loss2by4 + loss3by4)
            if self.use_EED:
                loss4by4 = self.kd_loss_function(output, temp) * (self.temperature**2)
                L_O += 0.1 * loss4by4

        elif self.loss_output == 'MSE':
            loss_mse_1 = MSEloss(middle_output1, target_output)
            loss_mse_2 = MSEloss(middle_output2, target_output)
            loss_mse_3 = MSEloss(middle_output3, target_output)
            L_O = loss_mse_1 + loss_mse_2 + loss_mse_3
            if self.use_EED:
                loss_mse_4 = MSEloss(output, target_output)
                L_O += loss_mse_4
        total_loss = L_C + L_O

        if self.use_feature_dist:
            feature_loss_1 = self.feature_loss_function(middle1_fea, target_fea)
            feature_loss_2 = self.feature_loss_function(middle2_fea, target_fea)
            feature_loss_3 = self.feature_loss_function(middle3_fea, target_fea)
            L_F = feature_loss_1 + feature_loss_2 + feature_loss_3
            if self.use_EED:
                feature_loss_4 = self.feature_loss_function(final_fea, target_fea)
                L_F += feature_loss_4
            total_loss += L_F

        return total_loss
    
    def kd_loss_function(self, output, target_output):
        """Compute kd loss"""
        """
        para: output: middle ouptput logits.
        para: target_output: final output has divided by temperature and softmax.
        """

        output = output / self.temperature
        output_log_softmax = torch.log_softmax(output, dim=1)
        loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
        return loss_kd

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss).mean()

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul(100.0 / batch_size))

        return res

    def validate(self, val_loader, model, device):
        model.eval()
        top1_acc = 0
        avg_acc = 0
        for (x,y) in val_loader:
            x = x.to(device)
            y = y.to(device)
            target = y
            with torch.no_grad():
                _ = model(x)
                output, middle_output1, middle_output2, middle_output3, \
                final_fea, middle1_fea, middle2_fea, middle3_fea = model.intermediary_output_list
                loss = self.criterion(output, target)
                prec1 = self.accuracy(output.data, target, topk=(1,))
                middle1_prec1 = self.accuracy(middle_output1.data, target, topk=(1,))
                middle2_prec1 = self.accuracy(middle_output2.data, target, topk=(1,))
                middle3_prec1 = self.accuracy(middle_output3.data, target, topk=(1,))
                top1_acc += prec1[0]
                avg_acc += (prec1[0]+middle1_prec1[0]+middle2_prec1[0]+middle3_prec1[0])/4
        top1_acc /= len(val_loader)
        avg_acc /= len(val_loader)
        model.train()
        return avg_acc, top1_acc 