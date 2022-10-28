import os
import psutil

def get_ram(strt='RAM: ', end='\n'):
    process = psutil.Process(os.getpid())
    print(strt+f'{int(process.memory_info().rss/10**6)} MB', end=end)