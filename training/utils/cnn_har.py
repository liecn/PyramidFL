# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class classificationModel(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim_in=1,conv_dim_out=5,conv_dim_conv=299*8):
        super(classificationModel, self).__init__()

        # self.conv1 = nn.Conv1d(conv_dim_in, 8, 5, stride=3)

        # self.dense = nn.Linear(conv_dim_conv, conv_dim_out*32)

        # self.lstm = nn.LSTM(
        #     # 8 * opts.image_height + opts.embedding,
        #     # 8 * int(opts.stft_nfft/2+1),
        #     900,
        #     conv_dim_out*32,
        #     batch_first=True,
        #     bidirectional=False)
        # self.fcn1 = nn.Linear(conv_dim_lstm*4, 300)

        self.fcn1 = nn.Linear(900, 300)
        self.fcn2 = nn.Linear(300, conv_dim_out)
        
        # self.softmax=nn.Softmax(dim=1).double()
        self.drop1 = nn.Dropout(0.2)
        self.act = nn.ReLU()

    def forward(self, x):
        # out=x.float().unsqueeze(1)
        
        # out = self.act(self.conv1(out))
        # out = out.reshape(out.size(0), -1)
        # out=self.act(self.dense(out))
        # out=self.drop2(out)

        # out,_ = self.lstm(out)
        # out = out.reshape(out.size(0), -1)

        out=x.float()
        out = self.act(self.fcn1(out))
        out=self.drop1(out)

        # out = self.softmax(self.fcn2(out))
        out = self.fcn2(out)
        return out
