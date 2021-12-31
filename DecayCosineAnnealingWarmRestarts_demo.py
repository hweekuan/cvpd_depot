import torch
import torch.optim as optim
import torch.nn    as nn

from DecayCosineAnnealingWarmRestarts import DecayCosineAnnealingWarmRestarts

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1,1)
    def forward(self,x):
        return self.fc(x)

if __name__=='__main__':

    net = model()
    lr = 1e-2
    step = 50
    decay = 0.9

    opt = optim.Adam(net.parameters(),lr)
    #print('opt ',opt)
    sch = DecayCosineAnnealingWarmRestarts(opt,step,decay)   

    for e in range(200):
        print(e,' ',opt.param_groups[0]['lr'])
        sch.step()




