nohup python diffpool_model.py 5000 data_8size3T_2.pkl diffpool3_model_8size_3classes_ver3 > model/log/diffpool3_model_8size_3classes_ver3.log 2>&1 &

nohup python topK_model.py 5000 data_8size3T.pkl topk_model_8size_3classes > model/log/topk_model_8size_3classes.log 2>&1 &

nohup python mincut_pool_model.py 5000 data_8size3T.pkl mincut_model_8size_3classes > model/log/mincut_model_8size_3classes.log 2>&1 &

ps aux | grep diffpool_model.py
ps aux | grep topK_model.py
ps aux | grep mincut_pool_model.py