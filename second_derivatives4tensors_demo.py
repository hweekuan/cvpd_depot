import torch
import numpy as np

# an example code of how to do second derivatives on a 
# multi-variate vector valued function
#
# test function is given by
#   R(q) = [a*q0,b*q0*(q1+1),c*q1*(q0*q0+q1)] - we call n-component of R as nparticle
#
# make dRdq as a tensor of shape [batch,nparticle=3,ndim=2]
# index as [b,i,j] -> dRidqj
#  dRdq[b,0,0] = dR0dq0 = a          dRdq[b,0,1] = dR0dq1 = 0
#  dRdq[b,1,0] = dR1dq0 = b*(q1+1)   dRdq[b,1,1] = dR1dq1 = b*q0
#  dRdq[b,2,0] = dR2dq0 = 2*c*q0*q1  dRdq[b,2,1] = dR2dq1 = c*q0*q0 + 2*c*q1
#
# the second derivative object is a tensor of shape [batch,nparticle,ndim,ndim]
# index as [b,i,j,k] -> dRidqjdqk =  d(dRidqj)/dqk
#  d2Rdq2[b,0,0,0] = d(R0dq0)dq0 = 0         d2Rdq2[b,0,0,1] = d(R0dq0)dq1 = 0
#  d2Rdq2[b,0,1,0] = d(R0dq0)dq0 = 0         d2Rdq2[b,0,1,1] = d(R0dq0)dq1 = 0
#  d2Rdq2[b,1,0,0] = d(R0dq0)dq0 = 0         d2Rdq2[b,1,0,1] = d(R0dq0)dq1 = b
#  d2Rdq2[b,1,1,0] = d(R0dq0)dq0 = b         d2Rdq2[b,1,1,1] = d(R0dq0)dq1 = 0
#  d2Rdq2[b,2,0,0] = d(R0dq0)dq0 = 2*c*q1    d2Rdq2[b,2,0,1] = d(R0dq0)dq1 = 2*c*q0
#  d2Rdq2[b,2,1,0] = d(R0dq0)dq0 = 2*c*q0    d2Rdq2[b,2,1,1] = d(R0dq0)dq1 = 2*c
#
# now we use autograd to take first and second derivative and compare
# autograd answers to hand differentiation answers. This will verify 
# that we do our autograd correctly
#
# R is written as a class with methods
# def __init__(self,[a,b,c])  pass in the coefficients for polynomial
# def __call__(self,q)        returns R(q), q.shape = [batch,ndim]
# def dRidqj(self,q)          returns the hand derived first derivative
# def d2Ridqjdqk(self,q)      returns the hand derived second derivative
# def auto_dRidqj(self,q)     returns autograd for first derivative
# def auto_d2Ridqjdqk(self,q) returns autograd for second derivative
# 

class R:

    def __init__(self,abc_list):
        self.a = abc_list[0]
        self.b = abc_list[1]
        self.c = abc_list[2]
    # =======================================
    #   R(q) = [a*q0,b*q0*(q1+1),c*q1*(q0*q0+q1)] - we call n-component of R as nparticle
    # q.shape = [batch,ndim]
    def __call__(self,q):
        t1 = self.a*q[:,0]
        t2 = self.b*q[:,0]*(q[:,1]+1)
        t3 = self.c*q[:,1]*(q[:,0]*q[:,0]+q[:,1])
        r = torch.stack((t1,t2,t3),dim=1)
        return r

    # =======================================
    # q.shape = [batch,ndim]
    # def dRidqj(self,q)          returns the hand derived first derivative
    def dRidqj(self,q):
        batch = q.shape[0]
        ndim = q.shape[1]
        dR0dq0 = torch.tensor([self.a]).repeat(batch,1)
        dR0dq1 = torch.zeros([batch,1])
        dR1dq0 = (self.b*(q[:,1]+1)).unsqueeze(1)
        dR1dq1 = (self.b*q[:,0]).unsqueeze(1)
        dR2dq0 = (2*self.c*q[:,0]*q[:,1]).unsqueeze(1)
        dR2dq1 = (self.c*q[:,0]*q[:,0]+2*self.c*q[:,1]).unsqueeze(1)

        dR0dq = torch.cat((dR0dq0,dR0dq1),dim=1).unsqueeze(1)
        dR1dq = torch.cat((dR1dq0,dR1dq1),dim=1).unsqueeze(1)
        dR2dq = torch.cat((dR2dq0,dR2dq1),dim=1).unsqueeze(1)

        dRdq = torch.cat((dR0dq,dR1dq,dR2dq),dim=1)
        return dRdq  # shape [batch,npar,ndim] = [b,i,j] indices

    # =======================================
    # q.shape = [batch,ndim]
    # def d2Ridqjdqk(self,q)      returns the hand derived second derivative
    def d2Ridqjdqk(self,q):

        dR0dq0dq0 = torch.zeros([batch,1])
        dR0dq0dq1 = torch.zeros([batch,1])
        dR0dq1dq0 = torch.zeros([batch,1])
        dR0dq1dq1 = torch.zeros([batch,1])
        dR1dq0dq0 = torch.zeros([batch,1])
        dR1dq0dq1 = torch.tensor([self.b]).repeat(batch,1)
        dR1dq1dq0 = torch.tensor([self.b]).repeat(batch,1)
        dR1dq1dq1 = torch.zeros([batch,1])
        dR2dq0dq0 = (2*self.c*q[:,1]).unsqueeze(1)
        dR2dq0dq1 = (2*self.c*q[:,0]).unsqueeze(1)
        dR2dq1dq0 = (2*self.c*q[:,0]).unsqueeze(1)
        dR2dq1dq1 = torch.tensor([2*self.c]).repeat(batch,1)
        
        dR0dq0dq = torch.cat((dR0dq0dq1,dR0dq0dq1),dim=1).unsqueeze(1)
        dR0dq1dq = torch.cat((dR0dq1dq1,dR0dq1dq1),dim=1).unsqueeze(1)
        dR1dq0dq = torch.cat((dR1dq0dq0,dR1dq0dq1),dim=1).unsqueeze(1)
        dR1dq1dq = torch.cat((dR1dq1dq0,dR1dq1dq1),dim=1).unsqueeze(1)
        dR2dq0dq = torch.cat((dR2dq0dq0,dR2dq0dq1),dim=1).unsqueeze(1)
        dR2dq1dq = torch.cat((dR2dq1dq0,dR2dq1dq1),dim=1).unsqueeze(1)

        dR0dqdq = torch.cat((dR0dq0dq,dR0dq1dq),dim=1).unsqueeze(1)
        dR1dqdq = torch.cat((dR1dq0dq,dR1dq1dq),dim=1).unsqueeze(1)
        dR2dqdq = torch.cat((dR2dq0dq,dR2dq1dq),dim=1).unsqueeze(1)

        d2Rdq2 = torch.cat((dR0dqdq,dR1dqdq,dR2dqdq),dim=1)

        return d2Rdq2  # shape [batch,npar,ndim,ndim] = [b,i,j,k] indices, dRidqjdqk

    # =======================================
    # q.shape = [batch,ndim]
    # def auto_dRidqj(self,q)     returns autograd for first derivative
    # indices goes as: dRidqj -> [b,i,j]
    def auto_dRidqj(self,q):
        r = self.__call__(q)
        ilist = []
        # I found that we need to extract tensor components (scalar) before
        # grad otherwise some components of grad subms up. don't know why
        for i in range(r.shape[1]):  # r.shape [batch,nparticle]
            value = torch.autograd.grad(r[:,i],q,grad_outputs=torch.ones_like(r[:,i]),create_graph=True)[0]
            ilist.append(value)           # value indices [b,j]   -> [batch,ndim]
        grad = torch.stack(ilist,dim=1)   # grad  indices [b,i,j] -> [batch,nparticle,ndim]
        return grad

    # =======================================
    # q.shape = [batch,ndim]
    # def auto_d2Ridqjdqk(self,q) returns autograd for second derivative
    # indices goes as: d(dRidqk)/dqk -> [b,i,j,k]
    #
    def auto_d2Ridqjdqk(self,q):
        dridqj = self.auto_dRidqj(q)
        ilist = []
        # I found that we need to extract tensor components (scalar) before
        # grad otherwise some components of grad subms up. don't know why
        for i in range(dridqj.shape[1]):
            jlist = []
            for j in range(dridqj.shape[2]):
                value = torch.autograd.grad(dridqj[:,i,j],q,\
                                   grad_outputs=torch.ones_like(dridqj[:,i,j]),create_graph=True)[0]
                jlist.append(value)           # value.indices [b,k]     -> [batch,ndim]
            jgrad = torch.stack(jlist,dim=1)  # jgrad.indices [b,j,k]   -> [batch,nparticle,ndim]
            ilist.append(jgrad)
        grad = torch.stack(ilist,dim=1)       # grad.indices  [b,i,j,k] -> [batch,nparticle,nparticle,ndim]
        return grad

# =======================================

def testdRdq(r_obj,q):
    drdq = r_obj.dRidqj(q)
    auto_drdq = r_obj.auto_dRidqj(q)
    diff = torch.abs(drdq-auto_drdq)
    return torch.max(diff)
# =======================================
def testd2Rdq2(r_obj,q):
    d2rdq2 = r_obj.d2Ridqjdqk(q)
    auto_d2rdq2 = r_obj.auto_d2Ridqjdqk(q)
    diff = torch.abs(d2rdq2-auto_d2rdq2)
    return torch.max(diff)
# =======================================
def print_d2Rdq2(d2Rdq2):

    batch = d2Rdq2.shape[0]
    npar  = d2Rdq2.shape[1]
    ndim  = d2Rdq2.shape[2]
    ndim  = d2Rdq2.shape[3]
    for i in range(npar):
        for j in range(ndim):
            for k in range(ndim):
                print('d2Rdq2[:',i,j,k,']',d2Rdq2[:,i,j,k])
# =======================================

if __name__=='__main__':

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2437)
    ntest= 1000
    batch = 100
    ndim  = 2 # fixed 
  
    err1_list = []
    err2_list = []
    for t in range(ntest):
        q = torch.rand([batch,ndim],requires_grad=True)
        abc_list = np.random.rand(3)
        r_obj = R(abc_list)
        
        auto_d2rdq2 = r_obj.auto_d2Ridqjdqk(q)
        d2rdq2 = r_obj.d2Ridqjdqk(q)
        
        err1 = testdRdq(r_obj,q).item()    # returns the err btw autograd and hand derived derivatives
        err2 = testd2Rdq2(r_obj,q).item()  # returns the err btw autograd and hand derived derivatives
        assert (err1<1e-9),'error in first derivative'
        assert (err2<1e-9),'error in second derivative'
        err1_list.append(err1)
        err2_list.append(err2)
        if t%100==0:
            print(t,'cummulative max error first  derivative err',max(err1_list))
            print(t,'cummulative max error second derivative err',max(err2_list))

    print('max first  derivative err',max(err1_list))
    print('max second derivative err',max(err2_list))


