import torch
import torch.nn as nn
from efficient_kan.kan import KAN
from FourierKAN.fftKAN import NaiveFourierKANLayer


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.model = nn.Sequential(nn.Linear(in_features, 50),
                                   nn.Tanh(),
                                   nn.Linear(50, out_features))
        
    def forward(self, x):
        return self.model(x)
    
    def sample(self, x, len_sample = 1):

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)


        if len(x) > self.in_features:
            x = x[:self.in_features]

        output = x

        for _ in range(len_sample):
            with torch.no_grad():
                out_step = self.forward(output[-self.in_features : ])
            output = torch.cat((output, out_step), dim=0)

        return output



# --------------------------------------------------------------------------------------------------------------------------- #



class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
    def sample(self, x, len_sample = 1):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() == 2:
            x = x.unsqueeze(0)

        output = x

        for _ in range(len_sample):
            out_step = self.forward(output[:, -x.size(1):, :])
            out_step = out_step.unsqueeze(1)
            output = torch.cat((output, out_step), dim=1)

        return output



# --------------------------------------------------------------------------------------------------------------------------- #



class KAN_temp(KAN):

    def sample(self, x, len_sample = 1):


        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)


        if len(x) > self.layers[0].in_features:
            x = x[:self.layers[0].in_features]

        output = x

        for _ in range(len_sample):
            with torch.no_grad():
                out_step = self.forward(output[-self.layers[0].in_features : ])
            output = torch.cat((output, out_step), dim=0)
        

        return output
    


# --------------------------------------------------------------------------------------------------------------------------- #



class FKAN(nn.Module):
    def __init__(self, features = [200, 100, 1], gridsize = 5, smooth_initialization = True):
        super(FKAN, self).__init__()
        self.in_features = features[0]
        self.layer = nn.Sequential(NaiveFourierKANLayer(features[0], features[1], gridsize = gridsize, smooth_initialization=smooth_initialization),
                                   NaiveFourierKANLayer(features[1], features[2], gridsize = gridsize, smooth_initialization=smooth_initialization))
        
    
    def forward(self, x):
        return self.layer(x)
    

    def sample(self, x, len_sample = 1):


        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)


        if len(x) > self.in_features:
            x = x[:self.in_features]

        output = x

        for _ in range(len_sample):
            with torch.no_grad():
                out_step = self.forward(output[-self.in_features : ])
            output = torch.cat((output, out_step), dim=0)
        

        return output