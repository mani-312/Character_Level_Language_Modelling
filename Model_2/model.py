import torch
import torch.nn as nn


class EncoderModel(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 100,
            vocab_size: int = 61,
            context_length: int = 10,
            device: torch.device = torch.device('cpu'),
    ) -> None:
        super(EncoderModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc_list = nn.ModuleList()
        for i in range(context_length):
            self.fc_list.append(nn.Linear(hidden_dim, 4 * hidden_dim))
        self.out = nn.Linear(4 * hidden_dim, vocab_size)
    
    def forward(self, ar: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(ar.shape[0], 4 * self.hidden_dim).to(self.device)
        for i in range(len(self.fc_list)):
            x += self.fc_list[i](self.embedding(ar[:, i]))
        x = self.out(x)
        return x


class LSTMEncoderModel(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 300,
            vocab_size: int = 61,  # Number of unique characters
    ) -> None:
        super(LSTMEncoderModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(input_size= hidden_dim,hidden_size= hidden_dim, batch_first=True, num_layers=2)
        self.out = nn.Linear(hidden_dim, vocab_size)

    # Predict the next character at each time step
    def forward(self, ar: torch.Tensor,h0,c0) -> torch.Tensor:
        # ar.shape = (batch_size, context_length)
        x = self.embedding(ar) # shape = (batch_size, context_length, hidden_dim)
        x, (hn,cn) = self.lstm(x,(h0,c0)) # x.shape = (batch_size, context_length, hidden_dim)

        x = x.reshape(-1,self.hidden_dim) # shape = (batch_size*context_length, hidden_dim)
        x = self.out(x) # shape = (batch_size*cotext_length, vocab_size)
        return x,(hn,cn)
    
    # This is best practice to initialize hidden states.(passing device as arguement).
    # Even after loading the saved model into device different from device used during training,
    #  the moment we perform init_hidden() all variables will be on the same device
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
