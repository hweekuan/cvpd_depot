# make this into a singleton
import torch

# copyright: code created by Lee Hwee Kuan 04September2021
# 
# !! please test code, code not tested with cuda yet
#

class mydevice(object):

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not mydevice.__instance:
            mydevice.__instance = object.__new__(cls)
        return mydevice.__instance

    def __init__(self):
        use_cuda = torch.cuda.is_available()
        mydevice.__instance.value = torch.device("cuda" if use_cuda else "cpu")
        print('device singleton constructed for ',mydevice.__instance.value)
        
    @staticmethod
    def load(x):
        return x.to(mydevice.__instance.value)

    @staticmethod
    def get():
        return mydevice.__instance.value
# ================================================


def verify_device(specified_device):

    target = torch.device(specified_device)

    if mydevice.get() != target: 
        print('WARNING: requested device unavailable', target)

    # this is to test if loading to GPU will throw an error
    a = torch.tensor([1])
    mydevice.load(a)


