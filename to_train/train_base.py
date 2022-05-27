import torch
from to_train.train_utils import get_device, validate_model, tab_str

# From https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
def train_single_epoch(model, data_loader, optimizer, loss_fn, device, dtype = torch.float32):
    running_loss = 0
    last_loss = 0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (x,y) in enumerate(data_loader):
        # Every data instance is an input + label pair
        x = x.to(device=device, dtype=dtype)  # move to device
        y = y.to(device=device, dtype=torch.long)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Compute the loss and its gradients
        loss = loss_fn(model,x, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        # For testing purposes 1000 --> 10
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

# From https://gitlab.doc.ic.ac.uk/lab2122_spring/DL_CW_1_lrc121/-/blob/master/dl_cw_1.ipynb
def train_loop(model, optimizer, scheduler,  data_loaders, loss_fn, epochs=1, gpu = -1):
    """
    Train a model.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    - gpu: (Optional) -1 for cpu, >=0 for gpu
    """
    device = get_device(gpu)
    train_loader, val_loader = data_loaders
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        last_loss = train_single_epoch(model,train_loader,optimizer,loss_fn, device)
        val_metrics = validate_model(loss_fn, model, val_loader, gpu)
        # had issues with trn_metrics, removed
        print(tab_str(e, last_loss))
        print(tab_str('', 0.0, *val_metrics))
        scheduler.step()
    return model

