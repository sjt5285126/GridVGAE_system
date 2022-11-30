nohup python diffpool_model.py 5000 data_4size3T.pkl model_4size_3classes_ver2 > model/log/model_4size_3classes_ver2.log 2>&1 &
ps aux | grep diffpool_model.py