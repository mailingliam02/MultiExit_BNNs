import torch
import models
import to_train
import argparse
import types
import numpy as np
from torch import nn
from evaluate import evaluate
from hyperparameters import get_hyperparameters
from to_train.train_utils import get_device
import datasets

# Difference in Confidence (and how much more/less confident it is compared )
def compare_confidence():
    # Need to look into metrics to measure error estimation
    pass

class ResourceLoader():
    def __init__(self):
        # Specify Hyperparameters (maybe add command line compatibility?)
        self.hyperparameters = get_hyperparameters(self.fake_args())
        #hyperparameters["loaders"]["batch_size"] = (1,1,1)
        self.train_loader, self.val_loader, self.test_loader = datasets.get_dataloader(self.hyperparameters["loaders"])
        # Evaluate the Network on Test
        self.test_loss_fn = to_train.get_loss_function(self.hyperparameters["test_loss"])

    def get_model(self, model_num, model_type = "val"):
        if model_type == "val":
            path = "/vol/bitbucket/lrc121/ind_proj/MultiExit_BNNs/snapshots/"
            model_type = "best_val_model_"
            print("Testing ", model_type+model_num)
            model_state = path+model_type+model_num
            self.hyperparameters["network"]["load_model"] = model_state
            # Follow og for info on how to parallelize!
            model = models.get_network(self.hyperparameters["network"])
        elif model_type == "test":
            path = "/vol/bitbucket/lrc121/ind_proj/MultiExit_BNNs/snapshots/"
            model_type = "final_model_"
            print("Testing ", model_type+model_num)
            model_state = path+model_type+model_num
            self.hyperparameters["network"]["load_model"] = model_state
            # Follow og for info on how to parallelize!
            model = models.get_network(self.hyperparameters["network"])
        return model
    
    def get_loader(self):
        return self.test_loader
    
    def fake_args(self):
        args = types.SimpleNamespace(dropout_exit = False, dropout_p = 0.5,
            dropout_type = None, n_epochs = 300, patience = 50)
        return args
    


class OverthinkingEvaluation():
    def __init__(self, model, test_loader, gpu=0, mc_dropout = False, mc_passes = 10):
        self.model = model
        self.test_loader = test_loader
        self.gpu = gpu
        self.mc_dropout = mc_dropout
        self.mc_passes = mc_passes
        self.layer_correct, self.layer_wrong, self.layer_predictions, \
            self.layer_confidence = self.sdn_get_detailed_results(self.model, 
            self.test_loader, gpu=self.gpu, mc_dropout = self.mc_dropout, mc_passes = self.mc_passes)

    # https://github.com/yigitcankaya/Shallow-Deep-Networks/blob/1719a34163d55ff237467c542db86b7e1f9d7628/model_funcs.py#L156
    def sdn_get_detailed_results(self, model, loader, gpu=0, mc_dropout = False, mc_passes = 10):
        """
        Returns:
        layer_correct : dict
            Each entry is a layer number containing the set of correctly
            classified instances 
        layer_wrong : dict
            Each entry is a layer number containing the set of incorrectly
            classified instances 
        layer_predictions : dict
            Each entry is a layer number containing a dictionary with the
            associated predictions for each instance
        layer_confidence : dict
            Each entry is a layer number containing a dictionary with the
            associated confidence of each prediction for each instance
        """
        device = get_device(gpu)
        model.eval()
        layer_correct = {}
        layer_wrong = {}
        layer_predictions = {}
        layer_confidence = {}

        outputs = list(range(model.n_exits))

        for output_id in outputs:
            layer_correct[output_id] = set()
            layer_wrong[output_id] = set()
            layer_predictions[output_id] = {}
            layer_confidence[output_id] = {}

        with torch.no_grad():
            for cur_batch_id, batch in enumerate(loader):
                b_x = batch[0].to(device)
                b_y = batch[1].to(device)
                if mc_dropout:
                    holder = np.empty((mc_passes,len(outputs),b_x.shape[0],model.out_dim))
                    for i in range(mc_passes):
                        output = model(b_x)
                        output_sm = [nn.functional.softmax(out, dim=1).cpu().numpy() for out in output]
                        output_sm_np = np.asarray(output_sm)
                        holder[i,:] = output_sm_np
                    output_sm_np = np.average(holder,axis = 0).squeeze()
                    output_sm = []
                    for i in range(output_sm_np.shape[0]):
                        output_sm.append(torch.from_numpy(output_sm_np[i]))
                else:
                    output = model(b_x)
                    output_sm = [nn.functional.softmax(out, dim=1) for out in output]
                for output_id in outputs:
                    cur_output = output[output_id]
                    cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]
                    pred = cur_output.max(1, keepdim=True)[1]
                    is_correct = pred.eq(b_y.view_as(pred))
                    for test_id in range(len(b_x)):
                        cur_instance_id = test_id + cur_batch_id*loader.batch_size
                        correct = is_correct[test_id]
                        layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                        layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                        if correct == 1:
                            layer_correct[output_id].add(cur_instance_id)
                        else:
                            layer_wrong[output_id].add(cur_instance_id)

        return layer_correct, layer_wrong, layer_predictions, layer_confidence

    def theoretical_best_performance(self):
        # No confidence exiting performance
        layers = sorted(list(self.layer_correct.keys()))
        end_correct = self.layer_correct[layers[-1]]
        end_wrong = self.layer_wrong[layers[-1]]
        total_number_of_samples = len(end_wrong)+len(end_correct)
        accuracy = len(end_correct)/total_number_of_samples
        # Confidence based exiting
        cum_correct = set()
        for layer in layers:
            cur_correct = self.layer_correct[layer]
            cum_correct = cum_correct | cur_correct
        theoretical_best_acc = len(cum_correct)/total_number_of_samples
        print(f"Current Final Exit Accuracy: {accuracy}")
        print(f"Theoretical Best Final Accuracy: {theoretical_best_acc}")
        return None

    def actual_best_performance(self, confidence_list = [0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]):
        for confidence_threshold in confidence_list:
            self.best_performance_with_confidence_threshold(confidence_threshold)
        return None

    def best_performance_with_confidence_threshold(self, confidence_threshold):
        layers = sorted(list(self.layer_correct.keys()))
        instances = set(list(self.layer_predictions[layers[0]].keys()))
        end_correct = self.layer_correct[layers[-1]]
        end_wrong = self.layer_wrong[layers[-1]]
        total_number_of_samples = len(end_wrong)+len(end_correct)
        final_correct = set()
        final_wrong = set()
        seen = set()
        for layer in layers:
            for instance in instances:
                if self.layer_confidence[layer][instance].item() > confidence_threshold:
                    seen.add(instance)
                    if self.layer_predictions[layer][instance].item() in self.layer_correct[layer]:
                        final_correct.add(instance)
                    # Add to predicted set
                    else:
                        final_wrong.add(instance)
                elif layers[-1] == layer:
                    seen.add(instance)
                    if self.layer_predictions[layer][instance].item() in self.layer_correct[layer]:
                        final_correct.add(instance)
                    # Add to predicted set
                    else:
                        final_wrong.add(instance)
            # remove seen instances
            instances = instances.difference(seen)
        assert len(instances) == 0
        assert len(final_correct)+len(final_wrong) == total_number_of_samples
        accuracy = len(final_correct)/total_number_of_samples
        print(f"Actual Best Accuracy with confidence threshold {confidence_threshold}: {accuracy}")
        return None

    # To quantify the wasteful effect of overthinking
    def wasteful_overthinking_experiment(self):
        layers = sorted(list(self.layer_correct.keys()))

        end_correct = self.layer_correct[layers[-1]]
        total = 10000

        # to quantify the computational waste (see Results for FLOP breakdown)
        c_i = [0.0614224109,0.1288826751,0.2280939973,0.2845412171,0.3630145257,
                0.4635139233,0.6145380774,0.693011386,0.7935107836,0.9160362701,1]
        #[,0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
        total_comp = 0

        cum_correct = set()
        for layer in layers:
            cur_correct = self.layer_correct[layer]
            unique_correct = cur_correct - cum_correct
            cum_correct = cum_correct | cur_correct

            print('Output: {}'.format(layer))
            print('Current correct: {}'.format(len(cur_correct)))
            print('Cumulative correct: {}'.format(len(cum_correct)))
            print('Unique correct: {}\n'.format(len(unique_correct)))

            if layer < layers[-1]:
                    total_comp += len(unique_correct) * c_i[layer]
            else:
                    total_comp += total - (len(cum_correct) - len(unique_correct))
            
        print('Total Comp: {}'.format(total_comp))
        return None

    # To quantify the destructive effects of overthinking
    def destructive_overthinking_experiment(self):
        layers = sorted(list(self.layer_correct.keys()))

        end_wrong = self.layer_wrong[layers[-1]]
        cum_correct = set()

        for layer in layers:
            cur_correct = self.layer_correct[layer]
            cum_correct = cum_correct | cur_correct
            cur_overthinking = cur_correct & end_wrong
            
            print('Output: {}'.format(layer))
            print('Current correct: {}'.format(len(cur_correct)))
            print('Cumulative correct: {}'.format(len(cum_correct)))
            print('Cur cat. overthinking: {}\n'.format(len(cur_overthinking)))

            total_confidence = 0.0
            for instance in cur_overthinking:
                    total_confidence += self.layer_confidence[layer][instance]
            
            print('Average confidence on destructive overthinking instances:{}'.format(total_confidence/(0.1 + len(cur_overthinking))))
            
            total_confidence = 0.0
            for instance in cur_correct:
                    total_confidence += self.layer_confidence[layer][instance]
            
            print('Average confidence on correctly classified :{}'.format(total_confidence/(0.1 + len(cur_correct))))
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adding dropout")
    parser.add_argument('--model_num', type=int, default=75)
    parser.add_argument('--model_type', type=str, default='val')
    parser.add_argument('--dropout', type = bool, default = False)
    parser.add_argument('--num_passes', type=int, default=10)
    parser.add_argument('--overthinking', type=bool,default = False)
    args = parser.parse_args()
    resources = ResourceLoader()
    test_loader = resources.get_loader()
    model = resources.get_model(args.model_num, model_type = "val")
    experiment = OverthinkingEvaluation(model, test_loader, gpu = 0, 
        mc_dropout = args.mc_dropout, mc_passes = args.num_passes)
    experiment.theoretical_best_performance()
    experiment.actual_best_performance()
    if args.overthinking:
        experiment.destructive_overthinking_experiment()
        experiment.wasteful_overthinking_experiment()

