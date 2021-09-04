# make this into a singleton
from mydevice import mydevice
from mydevice import verify_device

if __name__=='__main__':

    mydevice()

    verify_device('cpu')
    verify_device('cuda')

