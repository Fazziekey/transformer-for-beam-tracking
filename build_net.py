# Developer：Fazzie
# Time: 2020/12/721:30
# File name: build_net.py
# Development environment: Anaconda Python
import torch
from torch import nn

class transformer(nn.Module):
    def __init__(self,d_model,out_dim):

        super(transformer, self).__init__()
        self.d_model = d_model
        self.outdim = out_dim,  # 128

        # Define layers
        self.transform = nn.Transformer(
                d_model=d_model,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=2048,
                dropout=0.1,
                activation='relu',
                custom_encoder=None,
                custom_decoder=None
            )
        # self.classifier = nn.Linear(hid_dim,out_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  #  --> Softmax is implicitly implemented into the cross entropy loss

    def forward(self,encoder,decoder):
        out=self.transform(encoder,decoder)
        out = self.relu(out[:,-2:,:])
        # y = self.classifier(out)
        out = self.softmax(out)
        return out