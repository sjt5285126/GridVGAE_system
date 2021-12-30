from sys import argv

import dataset

if len(argv) < 5:
    print('python3 IsingInit.py size T_list nums name')
    exit()

size = int(argv[1])
T_list = eval(argv[2])
nums = int(argv[3])
name = argv[4]

dataset.IsingInit(size,T_list,nums,name)
