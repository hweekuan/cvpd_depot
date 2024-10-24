# copyright: code created by Lee Hwee Kuan 25August2021
# 
# this is a singleton class for printing system logs, pid, machine name, memory usage
# device name etc so that you can trace where, when, what resources are being used
# when you run your code. for a demo on how to use this code, look at system_logs_demo.py


import torch
import os
import platform
import numpy as np
import psutil
import time
from datetime import datetime

class system_logs(object):

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not system_logs.__instance:
            system_logs.__instance = object.__new__(cls)
        return system_logs.__instance

    def __init__(self,mydevice,snooze_time):
        system_logs.__instance.pid = os.getpid()
        system_logs.__instance.uname = os.uname()
        system_logs.__instance.mydevice = mydevice
        system_logs.__instance.start_time = datetime.now() #.strftime("%Y%m%d, %H:%M:%S")
        system_logs.__instance.start_time_sec = time.time() #.strftime("%Y%m%d, %H:%M:%S")
        system_logs.__instance.last_snooze_time = system_logs.__instance.start_time_sec 
        system_logs.__instance.snooze_time = snooze_time   # in seconds
        system_logs.__instance.cpu_memory_usage = []
        system_logs.__instance.cuda_memory_usage = []
        system_logs.__instance.start_cuda_memory = system_logs.__instance.current_cuda_memory()

    # record total memory use in current cuda device
    @staticmethod
    def current_cuda_memory():
        # print the cuda memory usage
        #free, total = torch.cuda.mem_get_info(system_logs.__instance.mydevice.get())
        free, total = torch.cuda.mem_get_info()
        total = total / 1024**2
        free  = free  / 1024**2
        mem_used_MB = (total - free)
        return mem_used_MB

    # print the pid,machine name,run start time
    # pid can be use to kill the job 
    @staticmethod
    def print_start_logs():
        print('pid : ', system_logs.__instance.pid)
        print('uname : ', system_logs.__instance.uname)
        print('code run start time ',system_logs.__instance.start_time.strftime("%Y%m%d, %H:%M:%S"))
        system_logs.__instance.mydevice.device_name()

    @staticmethod
    def record_memory_usage(t):
        mem_use = psutil.virtual_memory()[2]
        cur_mem =  system_logs.__instance.current_cuda_memory()
        cuda_mem = cur_mem-system_logs.__instance.start_cuda_memory
        system_logs.__instance.cpu_memory_usage.append(mem_use)
        system_logs.__instance.cuda_memory_usage.append(cuda_mem)
        print('cpu  memory usage :',mem_use,' at t=',t)
        print('cuda memory usage :',cuda_mem,' at t=',t)


    # check that if time has passed since last snooze time
    # this can be use to decide on code checkpoint, e.g. save model etc
    @staticmethod
    def check_alarm():
        now = time.time()
        run_duration = now - system_logs.__instance.last_snooze_time
        since_start = now - system_logs.__instance.start_time_sec
        if run_duration>system_logs.__instance.snooze_time:
            system_logs.__instance.last_snooze_time = now
            print('reset snooze time to be ',since_start)
            return True
        return False


    @staticmethod
    def print_current_logs():
        now = datetime.now()

        print('current date/time:', now.strftime("%Y%m%d, %H:%M:%S"))
        run_duration = now - system_logs.__instance.start_time
        print('run duration ',run_duration)

        mean_cpu_memory  = np.mean(system_logs.__instance.cpu_memory_usage)
        std_cpu_memory   = np.std(system_logs.__instance.cpu_memory_usage)
        mean_cuda_memory = np.mean(system_logs.__instance.cuda_memory_usage)
        std_cuda_memory  = np.std(system_logs.__instance.cuda_memory_usage)

        print('cpu  mean mem : ', mean_cpu_memory,  ', std mem : ', std_cpu_memory )
        print('cuda mean mem : ', mean_cuda_memory, ', std mem : ', std_cuda_memory )


