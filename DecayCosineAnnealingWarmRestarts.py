# written by Lee Hwee Kuan, 31Dec2021
#
# the problem with cosine annealing is that the lr oscillates and 
# at larger lr, it can throw off the optimiser. usually cosine lr 
# scheduler is use for small enough max lr. added a functionality 
# that have a decay max lr but operate like a cosine annealing lr scheme.


import torch.optim as optim

class DecayCosineAnnealingWarmRestarts:

    def __init__(self,optimizer,step_size,decay):

        self.opt = optimizer
        self.cosine_sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt,step_size)
        self.thrsh = optimizer.param_groups[0]['lr']
        self.step_size = step_size
        self.decay = decay
        self.cntr = 0


    def step(self):

        self.cntr += 1
        self.cosine_sch.step()
        cur_lr = self.opt.param_groups[0]['lr']
        
        if self.cntr%self.step_size==0:
            self.cntr = 0
            self.thrsh = self.thrsh*self.decay

        self.opt.param_groups[0]['lr'] = min(cur_lr,self.thrsh)



