from sys import argv

import dataset

if len(argv) == 1:
    print('python3 IsingInit.py size T_list nums')
    exit()

size = int(argv[1])
T_list = eval(argv[2])
nums = int(argv[3])

dataset.IsingInit(size,T_list,nums)
