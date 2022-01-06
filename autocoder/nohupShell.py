import os
from sys import argv

if len(argv) < 6:
    print("please input: python nohupShell.py size T_list nums name times")

size = argv[1]
T_list = argv[2]
nums = argv[3]
name = argv[4]
times = int(argv[5]) # 次数表示一共有多少个批次

for i in range(1,times+1):
    os.system('nohup python dataInit.py {0} {1} {2} {3}_{4} > sizeLog/{3}_{4}.log 2>&1 &'
              .format(size,T_list,nums,name,i))




