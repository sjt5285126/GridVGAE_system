import os
from sys import argv
import dataInit

size = int(argv[1])
T_list = eval(argv[2])
nums = int(argv[3])
name = argv[4]
time = int(argv[5]) # 次数表示一共有多少个批次

for i in range(1,time+1):
    os.system('nohup python dataInit.py {0} {1} {2} {3}_{4} > sizeLog/{3}_{4}.log 2>&1 &'
              .format(size,T_list,nums,name,i))




