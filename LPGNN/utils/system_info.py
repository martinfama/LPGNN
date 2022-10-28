import os
import psutil

def get_ram(strt='RAM: '):
    process = psutil.Process(os.getpid())
    print(strt+f'{process.memory_info().rss/10**6} MB')