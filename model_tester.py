import torch
import models
import to_train
import argparse
import types
import numpy as np
from torch import nn
from KDEpy import FFTKDE
from evaluate import evaluate
from hyperparameters import get_hyperparameters
from to_train.train_utils import get_device
import datasets
import torch
import numpy as np
import torch.nn.parallel

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

    def get_plot_data(self, dropout_bool):
        layers = sorted(list(self.layer_correct.keys()))
        instances = set(list(self.layer_predictions[layers[0]].keys()))
        for layer in layers:
            for instance in instances:
                if self.layer_predictions[layer][instance].item() in self.layer_correct[layer]:
                    if dropout_bool:
                        print(str(layer) + ",Correct,MC Dropout," + str(self.layer_confidence[layer][instance].item()))
                    else:
                        print(str(layer) + ",Correct,Standard," + str(self.layer_confidence[layer][instance].item()))
                else:
                    if dropout_bool:
                        print(str(layer) + ",Incorrect,MC Dropout," + str(self.layer_confidence[layer][instance].item()))
                    else:
                        print(str(layer) + ",Incorrect,Standard," + str(self.layer_confidence[layer][instance].item()))
        return None

# Almost all from https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/e41afbaf8181a0bd2fb194f9e9d30bcbe5b7f6c3/util_evaluation.py#L176
class UncertaintyQuantification():
    def __init__(self,model, test_loader, gpu = 0, 
        mc_dropout = False, mc_passes = 10):
        labels, p_evals, combined_p_evals = self.get_preds(model,test_loader, gpu = gpu, mc_dropout = mc_dropout, mc_passes = mc_passes)
        for exit_i in range(model.n_exits):
            ece, nll, mse, accu = self.ece_eval_binary(p_evals[exit_i], labels)
            print(f"ECE: {ece}, NLL: {nll}, MSE: {mse}, Accuracy: {accu}")
        ece, nll, mse, accu = self.ece_eval_binary(combined_p_evals, labels)
        print(f"ECE: {ece}, NLL: {nll}, MSE: {mse}, Accuracy: {accu}")

    def get_preds(self, model, loader, gpu=0, mc_dropout = False, mc_passes = 10):
        num_instances = len(loader.dataset)
        labels = np.zeros((num_instances,model.out_dim))
        preds = np.empty((model.n_exits,num_instances,model.out_dim))
        device = get_device(gpu)
        outputs = list(range(model.n_exits))
        model.eval()
        with torch.no_grad():
            for cur_batch_id, batch in enumerate(loader):
                b_x = batch[0].to(device)
                b_y = batch[1]
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
                    output_sm = [nn.functional.softmax(out, dim=1).cpu().numpy() for out in output]
                    output_sm_np = np.asarray(output_sm)
                for instance_id in range(b_x.shape[0]):
                    correct_output = b_y[instance_id].item()
                    labels[cur_batch_id*b_x.shape[0]+instance_id][correct_output] = 1
                # Will probably have issues if dataset size is not divisble by batch size
                preds[:,cur_batch_id*b_x.shape[0]:(1+cur_batch_id)*b_x.shape[0], :] = output_sm_np
                # Get combined average of each p
                combined_preds = np.average(preds, axis = 0)
        return labels, preds, combined_preds

    def mirror_1d(self, d, xmin=None, xmax=None):
        """If necessary apply reflecting boundary conditions."""
        if xmin is not None and xmax is not None:
            xmed = (xmin+xmax)/2
            return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
        elif xmin is not None:
            return np.concatenate((2*xmin-d, d))
        elif xmax is not None:
            return np.concatenate((d, 2*xmax-d))
        else:
            return d

    def ece_kde_binary(self, p,label,p_int=None,order=1):

        # points from numerical integration
        if p_int is None:
            p_int = np.copy(p)

        p = np.clip(p,1e-256,1-1e-256)
        p_int = np.clip(p_int,1e-256,1-1e-256)
        
        
        x_int = np.linspace(-0.6, 1.6, num=2**14)
        
        
        N = p.shape[0]

        # this is needed to convert labels from one-hot to conventional form
        label_index = np.array([np.where(r==1)[0][0] for r in label])
        with torch.no_grad():
            if p.shape[1] !=2:
                p_new = torch.from_numpy(p)
                p_b = torch.zeros(N,1)
                label_binary = np.zeros((N,1))
                for i in range(N):
                    pred_label = int(torch.argmax(p_new[i]).numpy())
                    if pred_label == label_index[i]:
                        label_binary[i] = 1
                    p_b[i] = p_new[i,pred_label]/torch.sum(p_new[i,:])  
            else:
                p_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
                label_binary = label_index
                    
        method = 'triweight'
        
        dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1)).numpy()
        kbw = np.std(p_b.numpy())*(N*2)**-0.2
        kbw = np.std(dconf_1)*(N*2)**-0.2
        # Mirror the data about the domain boundary
        low_bound = 0.0
        up_bound = 1.0
        dconf_1m = self.mirror_1d(dconf_1,low_bound,up_bound)
        # Compute KDE using the bandwidth found, and twice as many grid points
        pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
        pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        pp1 = pp1 * 2  # Double the y-values to get integral of ~1
        
        
        p_int = p_int/np.sum(p_int,1)[:,None]
        N1 = p_int.shape[0]
        with torch.no_grad():
            p_new = torch.from_numpy(p_int)
            pred_b_int = np.zeros((N1,1))
            if p_int.shape[1]!=2:
                for i in range(N1):
                    pred_label = int(torch.argmax(p_new[i]).numpy())
                    pred_b_int[i] = p_int[i,pred_label]
            else:
                for i in range(N1):
                    pred_b_int[i] = p_int[i,1]

        low_bound = 0.0
        up_bound = 1.0
        pred_b_intm = self.mirror_1d(pred_b_int,low_bound,up_bound)
        # Compute KDE using the bandwidth found, and twice as many grid points
        pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
        pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        pp2 = pp2 * 2  # Double the y-values to get integral of ~1

        
        if p.shape[1] !=2: # top label (confidence)
            perc = np.mean(label_binary)
        else: # or joint calibration for binary cases
            perc = np.mean(label_index)
                
        integral = np.zeros(x_int.shape)
        reliability= np.zeros(x_int.shape)
        for i in range(x_int.shape[0]):
            conf = x_int[i]
            if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
                accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
                if np.isnan(accu)==False:
                    integral[i] = np.abs(conf-accu)**order*pp2[i]  
                    reliability[i] = accu
            else:
                if i>1:
                    integral[i] = integral[i-1]

        ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
        return np.trapz(integral[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])


    def ece_hist_binary(self,p, label, n_bins = 15, order=1):
        
        p = np.clip(p,1e-256,1-1e-256)
        
        N = p.shape[0]
        label_index = np.array([np.where(r==1)[0][0] for r in label]) # one hot to index
        with torch.no_grad():
            if p.shape[1] !=2:
                preds_new = torch.from_numpy(p)
                preds_b = torch.zeros(N,1)
                label_binary = np.zeros((N,1))
                for i in range(N):
                    pred_label = int(torch.argmax(preds_new[i]).numpy())
                    if pred_label == label_index[i]:
                        label_binary[i] = 1
                    preds_b[i] = preds_new[i,pred_label]/torch.sum(preds_new[i,:])  
            else:
                preds_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
                label_binary = label_index

            confidences = preds_b
            accuracies = torch.from_numpy(label_binary)


            x = confidences.numpy()
            x = np.sort(x,axis=0)
            binCount = int(len(x)/n_bins) #number of data points in each bin
            bins = np.zeros(n_bins) #initialize the bins values
            for i in range(0, n_bins, 1):
                bins[i] = x[min((i+1) * binCount,x.shape[0]-1)]
                #print((i+1) * binCount)
            bin_boundaries = torch.zeros(len(bins)+1,1)
            bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1,1)
            bin_boundaries[0] = 0.0
            bin_boundaries[-1] = 1.0
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            
            ece_avg = torch.zeros(1)
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Calculated |confidence - accuracy| in each bin
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                #print(prop_in_bin)
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin)**order * prop_in_bin
        return ece_avg

    def ece_eval_binary(self,p, label):
        mse = np.mean(np.sum((p-label)**2,1)) # Mean Square Error
        N = p.shape[0]
        p = np.clip(p,1e-256,1-1e-256)
        nll = -np.sum(label*np.log(p))/N # log_likelihood
        accu = (np.sum((np.argmax(p,1)-np.array([np.where(r==1)[0][0] for r in label]))==0)/p.shape[0]) # Accuracy
        ece = self.ece_hist_binary(p,label).cpu().numpy() # ECE
        # or if KDE is used
        ece = self.ece_kde_binary(p,label)   

        return ece, nll, mse, accu



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adding dropout")
    parser.add_argument('--model_num', type=str, default='75')
    parser.add_argument('--model_type', type=str, default='val')
    parser.add_argument('--dropout', type = bool, default = False)
    parser.add_argument('--num_passes', type=int, default=10)
    parser.add_argument('--confidence_exiting', type=bool, default = False)
    parser.add_argument('--overthinking', type=bool,default = False)
    parser.add_argument('--uq', type=bool, default = False)
    parser.add_argument('--plotting',type=bool,default = False)
    args = parser.parse_args()
    resources = ResourceLoader()
    test_loader = resources.get_loader()
    model = resources.get_model(args.model_num, model_type = "val")
    if args.confidence_exiting or args.overthinking or args.plotting:
        experiment = OverthinkingEvaluation(model, test_loader, gpu = 0, 
            mc_dropout = args.dropout, mc_passes = args.num_passes)
        if args.confidence_exiting:
            experiment.theoretical_best_performance()
            experiment.actual_best_performance()
        if args.overthinking:
            experiment.destructive_overthinking_experiment()
            experiment.wasteful_overthinking_experiment()
        if args.plotting:
            experiment.get_plot_data(args.dropout)
    if args.uq:
        UncertaintyQuantification(model, test_loader, gpu = 0, 
            mc_dropout = args.dropout, mc_passes = args.num_passes)



