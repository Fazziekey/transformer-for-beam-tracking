import time

from Config import configs
from DataLoader import Train_DataLoader, Val_DataLoader
from build_net import transformer
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# print(torch.__version__)


model=transformer(d_model=configs['embedding_dim'],out_dim=configs['beam_num'])


optimizer = torch.optim.SGD(model.parameters(), lr=configs['learning_rate'])
embeddind = nn.Embedding(configs['beam_num'] + 1, configs['embedding_dim'])
loss_func = torch.nn.MSELoss(reduction='sum')
configs['train_size'] = Train_DataLoader.__len__()
configs['test_size'] = Val_DataLoader.__len__()


def train(model, trainloader, valloader, optimizer):
    """

    :type optimizer: object
    """
    itr = 0
    running_train_loss = []
    running_trn_top_1 = []
    running_val_top_1 = []
    train_loss_ind = []
    val_acc_ind = []
    print("----------- start train -----------")
    t_start = time.time()
    for epoch in range(configs['epoch']):
        model.train()
        for i, batch in enumerate(trainloader):

            itr += 1
            data, label = embeddind(batch['data'].t()), embeddind(batch['label'].t())
            out = model(data, label)
            batch_loss = loss_func(out, label)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            out = out.view(configs['batch_size'], configs['pred_step'], configs['beam_num'])
            pred_beams = torch.argmax(out, dim=2)
            label = label.view(configs['batch_size'], configs['pred_step'])
            top_1_acc = torch.sum(torch.prod(pred_beams == label, dim=1, dtype=torch.float)) / label.shape[0]
            if np.mod(itr, configs['coll_cycle']) == 0:  # Data collection cycle
                running_train_loss.append(batch_loss.item())
                running_trn_top_1.append(top_1_acc.item())
                train_loss_ind.append(itr)
            if np.mod(itr, configs['display_freq']) == 0:  # Display frequency
                print(
                    'Epoch No. {0}--Iteration No. {1}-- batch loss = {2:10.9f} and Top-1 accuracy = {3:5.4f}'.format(
                        epoch + 1,
                        itr,
                        batch_loss.item(),
                        top_1_acc.item())
                )

        # Validation:
        # -----------
        if np.mod(itr, configs['val_freq']) == 0:
            model.eval()
            batch_score = 0
            with torch.no_grad():
                for i, batch in enumerate(valloader):
                    data, label = embeddind(batch['data'].t()), embeddind(batch['label'].t())
                    out = model(data, label)
                    batch_loss = loss_func(out, label)
                    pred_beams = torch.argmax(out, dim=2)
                    batch_score += torch.sum(torch.prod(pred_beams == label, dim=1, dtype=torch.float))
                running_val_top_1.append(batch_score.cpu().numpy() / configs['test_size'])
                val_acc_ind.append(itr)
                print("Valid Epoch {} Batch {} loss: {}".format(epoch + 1, i, batch_loss.item()))
                print('Validation-- Top-1 accuracy = {0:5.4f}'.format(
                    running_val_top_1[-1])
                )

    t_end = time.time()
    train_time = (t_end - t_start) / 60
    print('Training lasted {0:6.3f} minutes'.format(train_time))
    print('------------------------ Training Done ------------------------')

    train_info = {'train_loss': running_train_loss,
                  'train_top_1': running_trn_top_1,
                  'val_top_1': running_val_top_1,
                  'train_itr': train_loss_ind,
                  'val_itr': val_acc_ind,
                  'train_time': train_time}
    return [model, configs, train_info]


if __name__ == '__main__':
    model, configs, train_info = train(model, Train_DataLoader, Val_DataLoader, optimizer)

    # Plot progress:
    if configs['prog_plot']:
        configs['fig_c'] += 1
        plt.figure(configs['fig_c'])
        plt.plot(train_info['train_itr'], train_info['train_top_1'], '-or', label='Train top-1')
        plt.plot(train_info['val_itr'], train_info['val_top_1'], '-ob', label='Validation top-1')
        plt.xlabel('Training iteration')
        plt.ylabel('Top-1 accuracy (%)')
        plt.grid(True)
        plt.legend()
        plt.show()

    pd.savemat(configs['results_file'], train_info)
