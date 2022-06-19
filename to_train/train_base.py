import torch
from to_train.train_utils import get_device, validate_model, tab_str, predict, plot_loss

# From https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
def train_single_epoch(model, data_loader, optimizer, loss_fn, device, dtype = torch.float32, max_norm = 10):
    running_loss = 0
    last_loss = 0
    all_losses = []
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
        # Needed for training (Need to look into this more!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        if i % 200 == 199:
            last_loss = running_loss / 200 # loss per batch
            all_losses.append(last_loss)
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss, all_losses

# From https://gitlab.doc.ic.ac.uk/lab2122_spring/DL_CW_1_lrc121/-/blob/master/dl_cw_1.ipynb
def train_loop(model, optimizer, scheduler,  data_loaders, loss_fn, experiment_id, max_norm = 1, patience = 10, epochs=1, gpu = -1):
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
    # Sets to train mode
    model.train()
    best_val_loss = float("inf")
    counter = 0
    all_train_losses = []
    all_val_losses = []
    for e in range(epochs):
        last_loss, train_losses = train_single_epoch(model,train_loader,optimizer,loss_fn, device, max_norm = max_norm)
        val_loss = validate_model(loss_fn, model, val_loader, gpu)
        all_train_losses += train_losses
        all_val_losses.append(val_loss.cpu())
        print(f"epoch: {e}, loss: {tab_str(last_loss)}, val_loss: {tab_str(val_loss)}")
        # had issues with trn_metrics, remove
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model, "./MultiExit_BNNs/snapshots/best_val_model_"+str(experiment_id))
        else:
            counter += 1
            if counter > patience:
                break
        scheduler.step()
    plot_loss(all_train_losses,all_val_losses, experiment_id)
    return model

