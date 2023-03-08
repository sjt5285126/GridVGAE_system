#!/bin/bash


nohup python model_init_GCN.py 2000 modelGCN_32_T_PTP_alpha16 > modelGCN_32_T_PTP_alpha16.log 2>&1 &

ps aux | grep model_init_GCN.py
#
nohup python model_init_GAT.py 2000 modelGAT_32_T_PTP_alpha16 > modelGCN_32_T_PTP_alpha16.log 2>&1 &

ps aux | grep model_init_GAT.py
#
#nohup python model_init_pre2.py 2000 model_16_PTPQuick_alpha_1 > log/model_16_PTPQuick_alpha_1.log 2>&1 &
#
#ps aux | grep model_init_pre2.py
#
#nohup python model_init_IsingXYNew.py 2000 modelIsingXYXY16A0.32_0712 > datalog/modelIsingXYXY16A0.32_0712.datalog 2>&1 &
#
#ps aux | grep model_init_pre2.py
