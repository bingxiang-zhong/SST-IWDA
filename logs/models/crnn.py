import torch
import torch.nn as nn
import numpy as np


class CausCnnBlock(nn.Module):
    """ Function: Basic convolutional block
    """

    # expansion = 1
    def __init__(self, inplanes, planes, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=True, downsample=None):
        super(CausCnnBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.Bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.Bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.pad = padding
        self.use_res = use_res

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.Bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.Bn2(out)

        if self.use_res == True:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out




class CRNN(nn.Module):
    """ Proposed model
    """

    def __init__(self, cnn_in_dim = 18, cnn_dim = 64, res_Phi = 180, rnn_in_dim = 256, rnn_hid_dim = 256):
        super(CRNN, self).__init__()

        self.cnn_in_dim = cnn_in_dim
        self.cnn_dim = cnn_dim
        self.res_Phi = res_Phi
        res_flag = False
        self.cnn = nn.Sequential(
            CausCnnBlock(cnn_in_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(4, 1)),
            CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 1)),
            CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 1)),
            CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 1)),
            CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 5)),
        )


        self.rnn = torch.nn.GRU(input_size = rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=1,
                                batch_first=True, bias=True)

        self.rnn_fc = nn.Sequential(
            torch.nn.Linear(in_features = rnn_hid_dim, out_features = 512),  # ,bias=False
            nn.Tanh(),
        )
        self.ipd2xyz = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.ipd2xyz2 = nn.Linear(256, self.res_Phi)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fea = x
        nb, _, nf, nt = fea.shape
        fea_cnn = self.cnn(fea)  # (nb, nch, nf, nt)

        fea_rnn_in = fea_cnn.view(nb, -1, fea_cnn.size(3))  # (nb, nch*nf,nt)

        fea_rnn_in = fea_rnn_in.permute(0, 2, 1)  # (nb, nt, nfea)


        fea_rnn, _ = self.rnn(fea_rnn_in) # output instead of hidden state

        fea_rnn_fc = self.rnn_fc(fea_rnn)  # (nb, nt, 2nf)

        fea_rnn_fc = self.relu(self.ipd2xyz(fea_rnn_fc))
        fea_rnn_fc = self.sigmoid(self.ipd2xyz2(fea_rnn_fc))

        return fea_rnn_fc


class crnnFE(nn.Module):
    """ Feature extractor (from input to GRU output) """

    def __init__(self, cnn_in_dim=18, cnn_dim=64):
        super(crnnFE, self).__init__()
        self.cnn_in_dim = cnn_in_dim
        self.cnn_dim = cnn_dim
        res_flag = False

        # CNN Blocks
        self.cnn = nn.Sequential(
            CausCnnBlock(cnn_in_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(4, 1)),
            CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 1)),
            CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 1)),
            CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 1)),
            CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 5)),
        )

        # RNN Configuration
        rnn_in_dim = 256
        rnn_hid_dim = 256
        rnn_bdflag = False
        self.rnn_ndirection = 2 if rnn_bdflag else 1
        self.rnn = nn.GRU(
            input_size=rnn_in_dim,
            hidden_size=rnn_hid_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=rnn_bdflag
        )

    def forward(self, x):
        # CNN processing
        nb, _, nf, nt = x.shape
        fea_cnn = self.cnn(x)

        # Prepare RNN input
        fea_rnn_in = fea_cnn.view(nb, -1, fea_cnn.size(3)).permute(0, 2, 1)

        # RNN processing
        fea_rnn, fea_state = self.rnn(fea_rnn_in)
        fea_state = fea_state.squeeze(0)
        return fea_rnn, fea_state  # Output shape: (nb, nt, rnn_hid_dim * num_directions)


class Localizer(nn.Module):
    """ Localizer (after GRU) """

    def __init__(self, res_Phi=180):
        super(Localizer, self).__init__()
        self.res_Phi = res_Phi
        ratio = 2
        rnn_hid_dim = 256
        rnn_out_dim = 128 * 2 * ratio

        self.rnn_fc = nn.Sequential(
            nn.Linear(rnn_hid_dim, rnn_out_dim),  # Simplified since we know GRU output dims
            nn.Tanh()
        )
        self.ipd2xyz = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.ipd2xyz2 = nn.Linear(256, res_Phi)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is the output from FeatureExtractor (GRU output)
        x = self.rnn_fc(x)
        x = self.relu(self.ipd2xyz(x))
        x = self.sigmoid(self.ipd2xyz2(x))
        return x


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=7, python -m models.arch.SpatialNet
    x = torch.randn((1, 18, 257, 50))  # .cuda() # 251 = 4 second; 129 = 8 kHz; 257 = 16 kHz
    model = CRNN()
    y = model(x)
    print(y.shape)
