import time
import torch

import numpy as np

from os import path
from tqdm import trange

def train(model, iterator, optimizer, criterion, accuracy_fn, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    # has to be assigned here as some iterators
    # change length after each loop
    len_iterator = len(iterator)

    for (x, y) in iterator:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = accuracy_fn(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len_iterator, epoch_acc / len_iterator


def evaluate(model, iterator, criterion, accuracy_fn, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    len_iterator = len(iterator)

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            
            loss = criterion(y_pred, y)

            acc = accuracy_fn(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len_iterator, epoch_acc / len_iterator



def train_and_checkpoint(model, train_dataloader, test_dataloader, optimizer, criterion, accuracy_fn, device='cpu', num_epochs=75, path_to_save='checkpoints'):
    """
    Trains a model on `train_dataloader` and evaluates it on `test_dataloader`.
    Saves the model weights in a file named by formatting path_to_save (read Args below).


    Args:
        path_to_save (str): the path of the dir to save the checkpoints in the format
                            'ck-{i}.pt' where `i` is the number of epochs.

    returns:
        a dictionary containing the history of loss and scores and the time it
        took to train the model.
    """

    pbar = trange(num_epochs, desc='Training', position=0, leave=True)

    # Lists to keep track of train history
    train_losses, test_losses, train_scores, test_scores = [], [], [], []

    info = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'loss': np.Inf,
        'acc': np.NINF,
        'epochs': num_epochs,
        'time': 0
    }

    # Evaluate at inital weight
    train_loss, train_acc = evaluate(model, train_dataloader, criterion, accuracy_fn, device)    
    test_loss, test_acc = evaluate(model, test_dataloader, criterion, accuracy_fn, device)

    train_losses.append(train_loss), train_scores.append(train_acc)
    test_losses.append(test_loss), test_scores.append(test_acc)

    # Save model's initial weight
    torch.save(model.state_dict(), path.join(path_to_save, 'ck-0.pt'))
    torch.save(train_loss, path.join(path_to_save, 'loss-0.pt'))

    for epoch in pbar:

        start = time.time()
        train(model, train_dataloader, optimizer, criterion, accuracy_fn, device)
        stop = time.time()

        train_loss, train_acc = evaluate(
            model, train_dataloader, criterion, accuracy_fn, device)
        
        test_loss, test_acc = evaluate(
            model, test_dataloader, criterion, accuracy_fn, device)

        info['loss'] = test_loss
        info['acc'] = test_acc
        info['time'] = info['time'] + stop - start

        train_losses.append(train_loss), train_scores.append(train_acc)
        test_losses.append(test_loss), test_scores.append(test_acc)

        pbar.set_description(
            f'Test / Train | Loss: {test_loss:.3f}/{train_loss:.3f} | Acc: {test_acc*100:.2f}/{train_acc*100:.2f}')
        
        torch.save(model.state_dict(), path.join(path_to_save, f'ck-{epoch+1}.pt'))
        torch.save(train_loss, path.join(path_to_save, f'loss-{epoch+1}.pt'))

    pbar.close()
    return info