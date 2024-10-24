from system_logs import system_logs
from mydevice    import mydevice

import time
import torch

# =================================
class t2:
    def f(self,t):
        system_logs.record_memory_usage(t)
# =================================

if __name__=='__main__':

    _ = mydevice()

    # snooze time in units of seconds can be use to check how much time
    # your code has been running and to check point your code
    snooze_time = 2  

    # singleton object
    _ = system_logs(mydevice,snooze_time)

    # print out information of machine etc
    system_logs.print_start_logs()

    t = t2()

    for i in range(7):

        t.f(1)
        time.sleep(1)
       
        # check if it is time to checkpoint your code or not
        if system_logs.check_alarm():
            print('alarm sound, make snooze again')
            print('we can use this alarm clock to decide if we want to checkpoint our code')
       
        y = mydevice.load(torch.randn([10000,10000],requires_grad=True))
        w = torch.sum(y*y)
        w.backward()
       
        t.f(2)

        # print information of current running status
        system_logs.print_current_logs()   
        print('---------------------------------------')


