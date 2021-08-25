from system_logs import system_logs

import time
import torch

# =================================
class device:
    def __init__(self):
        pass
    def device_name(self):
        print('my device here')
# =================================
class t2:
    def f(self,t):
        system_logs.record_memory_usage(t)
# =================================

if __name__=='__main__':

    mydevice = device()

    _ = system_logs(mydevice)

    system_logs.print_start_logs()

    t = t2()
 
    t.f(1)
    time.sleep(1)

    y = torch.randn([10000,10000],requires_grad=True)
    w = torch.sum(y*y)
    w.backward()

    t.f(2)

    system_logs.print_end_logs()   


