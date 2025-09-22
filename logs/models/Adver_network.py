import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=1):
        super(DomainClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1,
                                batch_first=True, bias=True, dropout=0.0, bidirectional=False)
        layers = [
        nn.Linear(hidden_dim, 128),
        nn.ReLU(),
        nn.Linear(128, num_domains),
        nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, lengths):
        
        # Use packed sequence for efficiency
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, x = self.rnn(packed) #hidden state as feature
        x = x[-1]
        x = self.layers(x)
        return x