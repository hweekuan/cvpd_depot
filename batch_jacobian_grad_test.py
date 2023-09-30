import torch
import torch.autograd.functional as F
from batch_jacobian_grad import batch_jacobian
from batch_jacobian_grad import batch_jacobian_grad
# ------------------------------------------
class sqf:
    def __call__(self,x):
        t1 = x*x
        ret = torch.stack((t1[:,0]*x[:,1],t1[:,1]*x[:,0]*x[:,0]),dim=1)
        return ret
# ------------------------------------------
class Jsqf1:
    def __call__(self,x):
        df1dx1 = torch.unsqueeze(2*x[:,0]*x[:,1],dim=1)
        df1dx2 = torch.unsqueeze(x[:,0]**2,dim=1)
        df2dx1 = torch.unsqueeze(2*x[:,0]*x[:,1]**2,dim=1)
        df2dx2 = torch.unsqueeze(2*x[:,1]*x[:,0]**2,dim=1)
        
        print('df1dx1',df1dx1)
        print('df1dx2',df1dx2)
        print('df2dx1',df2dx1)
        print('df2dx2',df2dx2)
        
        Jf1 = torch.cat((df1dx1,df1dx2),dim=1)
        Jf2 = torch.cat((df2dx1,df2dx2),dim=1)
        J = torch.stack((Jf1,Jf2),dim=1)
        print('J',J)

        return J
# ------------------------------------------
class Jsqf2:
    def __call__(self,x):
        dJ11dx1 = torch.unsqueeze(2*x[:,1],dim=1)
        dJ12dx1 = torch.unsqueeze(2*x[:,0],dim=1)
        dJ21dx1 = torch.unsqueeze(2*x[:,1]**2,dim=1)
        dJ22dx1 = torch.unsqueeze(4*x[:,0]*x[:,1],dim=1)

        dJ11dx2 = torch.unsqueeze(2*x[:,0],dim=1)
        dJ12dx2 = torch.unsqueeze(0*x[:,0],dim=1)
        dJ21dx2 = torch.unsqueeze(4*x[:,0]*x[:,1],dim=1)
        dJ22dx2 = torch.unsqueeze(2*x[:,0]**2,dim=1)
        
        J1x1 = torch.cat((dJ11dx1,dJ12dx1),dim=1)
        J2x1 = torch.cat((dJ21dx1,dJ22dx1),dim=1)
        Jx1 = torch.stack((J1x1,J2x1),dim=1)

        J1x2 = torch.cat((dJ11dx2,dJ12dx2),dim=1)
        J2x2 = torch.cat((dJ21dx2,dJ22dx2),dim=1)
        Jx2 = torch.stack((J1x2,J2x2),dim=1)

        dJdx = torch.stack((Jx1,Jx2),dim=1)
        print('dJdx',dJdx)
        return dJdx
# ------------------------------------------
    
if __name__=='__main__':

    in_dim = 2
    out_dim = 2
    batch = 3

    torch.manual_seed(43821)
    c = sqf()
    jsqf1_obj = Jsqf1()
    jsqf2_obj = Jsqf2()
    j1_obj = batch_jacobian(c)
    j2_obj = batch_jacobian_grad(j1_obj)

    x = torch.rand([batch,in_dim],requires_grad=True)
    print('x',x)
    jsqf1 = jsqf1_obj(x)
    j1 = j1_obj(x,create_graph=True)
    diff1 = torch.sum(torch.abs(j1-jsqf1))
    assert diff1<1e-6,'error in jacobian'
    print('diff1',diff1)

    # visual check for indices
    print('indexing j1[b][0][1]',j1[:,0,1])
    print('indexing j1[b][1][0]',j1[:,1,0])
 
    jsqf2 = jsqf2_obj(x)
    j2 = j2_obj(x,create_graph=True)
    print('jsqf2',jsqf2)
    print('j2',j2)
    diff2 = torch.sum(torch.abs(j2-jsqf2))
    assert diff2<1e-6,'error in grad of jacobian'
    print('diff2',diff2)

    # visual check for indices
    print('indexing j2[b][j=0][k=0][i=0]',j2[:,0,0,0])
    print('indexing j2[b][j=0][k=0][i=1]',j2[:,0,0,1])
    print('indexing j2[b][j=0][k=1][i=0]',j2[:,0,1,0])
    print('indexing j2[b][j=1][k=0][i=0]',j2[:,1,0,0])
    print('indexing j2[b][j=1][k=0][i=1]',j2[:,1,0,1])
    print('indexing j2[b][j=1][k=1][i=0]',j2[:,1,1,0])

