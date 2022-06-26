import torch
import models
import to_train
import argparse
from torch import nn
from evaluate import evaluate
from hyperparameters import get_hyperparameters
from to_train.train_utils import get_device
import datasets


def load_model(hyperparameters, model_num, model_type = "val"):
    if model_type == "val":
        path = "/vol/bitbucket/lrc121/ind_proj/MultiExit_BNNs/snapshots/"
        model_type = "best_val_model_"
        print("Testing ", model_type+model_num)
        model_state = path+model_type+model_num
        hyperparameters["network"]["load_model"] = model_state
        # Follow og for info on how to parallelize!
        model = models.get_network(hyperparameters["network"])
    elif model_type == "test":
        path = "/vol/bitbucket/lrc121/ind_proj/MultiExit_BNNs/snapshots/"
        model_type = "final_model_"
        print("Testing ", model_type+model_num)
        model_state = path+model_type+model_num
        hyperparameters["network"]["load_model"] = model_state
        # Follow og for info on how to parallelize!
        model = models.get_network(hyperparameters["network"])
    return model

# Overthinking Analysis
def calculate_overthinking(model, test_iter, gpu):
    device = get_device(gpu)
    model.eval()
    for (x,y) in test_iter:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            output = model(x)


# https://github.com/yigitcankaya/Shallow-Deep-Networks/blob/1719a34163d55ff237467c542db86b7e1f9d7628/model_funcs.py#L156
def sdn_get_detailed_results(model, loader, gpu=0):
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

# To quantify the wasteful effect of overthinking
def wasteful_overthinking_experiment(model, test_loader, gpu=0):
    layer_correct, _, _, _ = sdn_get_detailed_results(model, loader=test_loader, gpu=gpu)
    layers = sorted(list(layer_correct.keys()))

    end_correct = layer_correct[layers[-1]]
    total = 10000

    # to quantify the computational waste (see Results for FLOP breakdown)
    c_i = [0.0614224109,0.1288826751,0.2280939973,0.2845412171,0.3630145257,
            0.4635139233,0.6145380774,0.693011386,0.7935107836,0.9160362701,1]
    #[,0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    total_comp = 0

    cum_correct = set()
    for layer in layers:
        cur_correct = layer_correct[layer]
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

# Difference in Confidence (and how much more/less confident it is compared )


# Performance with Confidence-Based Exiting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adding dropout")
    parser.add_argument('--dropout_exit', type=bool, default=False)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--dropout_type', type=str, default=None)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()
    # Specify Hyperparameters (maybe add command line compatibility?)
    hyperparameters = get_hyperparameters(args)

    train_loader, val_loader, test_loader = datasets.get_dataloader(hyperparameters["loaders"])
    # Evaluate the Network on Test
    test_loss_fn = to_train.get_loss_function(hyperparameters["test_loss"])

    # Load the Network
    print("Getting Network")
    # List of models to test
    model_list = ["75","76"]
    for model_num in model_list:
        model = load_model(hyperparameters, model_num, model_type = "val")
        #results = evaluate(test_loss_fn, test_loader,model,hyperparameters["gpu"], 0, hyperparameters["mc_dropout_passes"], create_log = False)
        wasteful_overthinking_experiment(model,test_loader,gpu=hyperparameters["gpu"])