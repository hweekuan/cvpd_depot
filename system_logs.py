# copyright: code created by Lee Hwee Kuan 25August2021
# 
# this is a singleton class for printing system logs, pid, machine name, memory usage
# device name etc so that you can trace where, when, what resources are being used
# when you run your code. for a demo on how to use this code, look at system_logs_demo.py


import os
import platform
import numpy as np
import psutil
from datetime import datetime

class system_logs(object):

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not system_logs.__instance:
            system_logs.__instance = object.__new__(cls)
        return system_logs.__instance

    def __init__(self,mydevice):
        system_logs.__instance.pid = os.getpid()
        system_logs.__instance.uname = os.uname()
        system_logs.__instance.mydevice = mydevice
        system_logs.__instance.start_time = datetime.now() #.strftime("%Y%m%d, %H:%M:%S")
        system_logs.__instance.memory_usage = []

    @staticmethod
    def print_start_logs():
        print('pid : ', system_logs.__instance.pid)
        print('uname : ', system_logs.__instance.uname)
        print('code run start time ',system_logs.__instance.start_time.strftime("%Y%m%d, %H:%M:%S"))
        system_logs.__instance.mydevice.device_name()

    @staticmethod
    def record_memory_usage(t):
        mem_use = psutil.virtual_memory()[2]
        system_logs.__instance.memory_usage.append(mem_use)
        print('memory usage :',mem_use,' at t=',t)

    @staticmethod
    def print_end_logs():
        now = datetime.now()

        print('end date/time every checkpoint:', now.strftime("%Y%m%d, %H:%M:%S"))
        run_duration = now - system_logs.__instance.start_time
        print('run time ',run_duration)

        mean_memory = np.mean(system_logs.__instance.memory_usage)
        std_memory = np.std(system_logs.__instance.memory_usage)
        print('mean mem : ', mean_memory, ', std mem : ', std_memory )


