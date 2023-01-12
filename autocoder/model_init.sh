#!/bin/bash


#nohup python model_init_GCN.py 500 modelGCN_32_T_PTP_0706 > datalog/modelGCN_32_T_PTP_0706.datalog 2>&1 &
#
#ps aux | grep model_init_GCN.py
#
nohup python model_init_pre2.py 2000 model_16_32_PTPQuick > log/model_16_32_PTPQuick.log 2>&1 &

ps aux | grep model_init_pre2.py

#nohup python model_init_IsingXYNew.py 2000 modelIsingXYXY16A0.32_0712 > datalog/modelIsingXYXY16A0.32_0712.datalog 2>&1 &

#ps aux | grep model_init_pre2.py
