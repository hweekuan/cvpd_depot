import torch
import torch.autograd.functional as F

# ------------------------------------------
# class that compute the jacobian
# of a function in which this function
# can take batch of data. input to function
# x.shape = [batch,dim], dim is the dimension 
# of vector x
# Jki = dfk/dxi, index as J[b][k][i] were b is the batch index
#
class batch_jacobian:
    def __init__(self,function):
        self.f = function
    def __call__(self,x,create_graph=True):
        J = F.jacobian(self.f,x,create_graph=create_graph)
        # because of batch, torch considers the whole batch
        # in autograd computations.
        J = torch.sum(J,dim=2) 
        return J
# ------------------------------------------
# computes the gradient of the jacobian matrix
# in which this jacobian matrix come as a batch
# dJki/dxj = d/dxj (dfk/dxi), index as dJ[b][j][k][i]
#
class batch_jacobian_grad:
    # J: the jacobian matrix index as J[b][k][i], Jki=dfk/dxi
    def __init__(self,J):
        self.J = J
    # gradient evaluated at x with x.shape=[batch,dim]
    def __call__(self,x,create_graph=True):
        dJ = F.jacobian(self.J,x,create_graph=create_graph)
        # because of batch, torch considers the whole batch
        # in autograd computations. hence the need to sum over
        dJ = torch.sum(dJ,dim=3)
        dJ = torch.permute(dJ,[0,2,1,3])
        return dJ
# ------------------------------------------
    
