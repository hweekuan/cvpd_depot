# written by Lee Hwee Kuan, 31Dec2021
#
# the problem with cosine annealing is that the lr oscillates and 
# at larger lr, it can throw off the optimiser. usually cosine lr 
# scheduler is use for small enough max lr. added a functionality 
# that have a decay max lr but operate like a cosine annealing lr scheme.

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
    step = 500
    decay = 0.9
    minlr = 1e-3

    opt = optim.Adam(net.parameters(),lr)
    #print('opt ',opt)
    sch = DecayCosineAnnealingWarmRestarts(opt,step,decay,minlr)   

    for e in range(2000):
        print(e,' ',opt.param_groups[0]['lr'])
        sch.step()




