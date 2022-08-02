import torch
import numpy as np


# returns the tensor contraction
# of two tensors with indices "dims"
# when the dimensions to be contracted
# are None, then function returns direct
# products of tensors
#
# x_dims is a list of dimensions to be contracted
# y_dims is a list of dimensions to be contracted
# e.g. 
# tensor_contraction(x,y,[0,2],[1,3])
# contracts indices of 0 and 1 as a pair
# contracts indices of 2 and 3 as a pair
#
def tensor_contraction(x,y,x_dims,y_dims):

    if x_dims==[] and y_dims==[]:
        x_dims2 = tuple(x.shape) + tuple(np.ones(len(y.shape),dtype=int))
        y_dims2 = tuple(np.ones(len(x.shape),dtype=int)) + tuple(y.shape)
        new_x = x.view(x_dims2)
        new_y = y.view(y_dims2)
        return new_x*new_y # broadcast

    return torch.tensordot(x,y,(x_dims,y_dims))
 

