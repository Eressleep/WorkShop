#from model import Model
import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.cuda.FloatTensor) 
import torch.nn.functional as F

dtype = torch.float32
input_dim = 48
hidden_dim = 100
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 1
criterion = nn.MSELoss()
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def trans(tens, x):
    new = torch.randn(1,1,48)
    list1 = list(tens[0][0][1:])
    list1.append(x)
    for i in range(len(tens[0][0])):
        new[0][0][i] = list1[i]
    return(new)


class NET(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NET, self).__init__()
     # Hidden dimensions
        self.hidden_dim = hidden_dim

    # Number of hidden layers
        self.layer_dim = layer_dim

    # Building your LSTM
    # batch_first=True causes input/output tensors to be of shape
    # (batch_dim, seq_dim, feature_dim)
        self.lstmCell = nn.LSTMCell(input_dim, hidden_dim)

    # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        return model

    
    def forward(self, x, hidden):
        ht1, ct1 = self.lstmCell(x, hidden)
        x = self.fc(ht1)
        return x, (ht1, ct1)



    def init_hidden(self, batch_size=1):
        return (torch.zeros(batch_size, self.hidden_dim, requires_grad=True).to(device),
                torch.zeros(batch_size, self.hidden_dim, requires_grad=True ).to(device))



    def predict(model, start, steps,hidden = model.init_hidden()):
        #Зададим начальные состояния
        dat = torch.zeros(1,1,48).to(device)
        dat[0][0] = start
        predict = [] 
        hidden = model.init_hidden()
        #Последовательность прогноза
        for i in range(steps-1):
            
            out ,hidden = model(dat,hidden)
            dat = trans(dat,float(out))
            predict.append(out)
        return(predict)

    


    
