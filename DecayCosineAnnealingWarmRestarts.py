# written by Lee Hwee Kuan, 31Dec2021
# written by Lee Hwee Kuan, 11Feb2022
#
# the problem with cosine annealing is that the lr oscillates and 
# at larger lr, it can throw off the optimiser. usually cosine lr 
# scheduler is use for small enough max lr. added a functionality 
# that have a decay max lr but operate like a cosine annealing lr scheme.

import torch
import torch.optim as optim

class DecayCosineAnnealingWarmRestarts:

    def __init__(self,optimizer,step_size,decay):

        self.opt = optimizer
        self.cosine_sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt,step_size)
        self.thrsh = optimizer.param_groups[0]['lr']
        self.step_size = step_size
        self.decay = decay
        self.cntr = 0

        print('state dict ',self.state_dict())


    def step(self):

        self.cntr += 1
        self.cosine_sch.step()
        cur_lr = self.opt.param_groups[0]['lr']
        
        if self.cntr%self.step_size==0:
            self.cntr = 0
            self.thrsh = self.thrsh*self.decay

        self.opt.param_groups[0]['lr'] = min(cur_lr,self.thrsh)


    def state_dict(self):
        return { "cos_dict" : self.cosine_sch.state_dict(), "thrsh" : self.thrsh, "cntr" : self.cntr }


    def load_state_dict(self,read_dict):
        self.cosine_sch.load_state_dict(read_dict['cos_dict'])
        self.thrsh = read_dict['thrsh']
        self.cntr = read_dict['cntr']




# for testing code
if __name__=='__main__':


    param = torch.rand([2],requires_grad=True)
    lr = 1
    opt = optim.SGD([param],lr)

    dca = DecayCosineAnnealingWarmRestarts(opt,10,.5)

    dca.step() # perform 3 steps and see how
    dca.step()
    dca.step()

    print('state dict before ',dca.state_dict())
    print('cos dict before ',dca.state_dict()['cos_dict'])

    torch.save({'sch': dca.state_dict()},'testfile.pth')

    ckpt = torch.load('testfile.pth')

    dca.load_state_dict(ckpt['sch'])

    print('========')
    print('state dict ',dca.state_dict())
    print('cos dict after ',dca.state_dict()['cos_dict'])




