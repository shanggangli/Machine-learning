def run_lgb(X,y):
    params = {'bagging_fraction': 0.9,
              'objective':'regression',
              'metric':'rmse',
            "verbosity" : -1,
              'bagging_seed':42,
                'bagging_freq': 10, 
                'feature_fraction': 0.8, 
                'learning_rate': 0.01, 
                'max_depth': 5, 
                'min_data_in_leaf': 8, 
                'num_leaves': 30, 
                'reg_alpha': 31, 
                'reg_lambda': 40}

        
    #part 3.交叉验证模型
    model_list = []
    train_pred = np.zeros(len(X))
    
    # Part 4.交叉验证
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    
    for train_part_index, eval_index in kf.split(X, y):
        # 训练数据封装
        train_part = lgb.Dataset(X.iloc[train_part_index],
                                 y.iloc[train_part_index])
        # 测试数据封装
        eval = lgb.Dataset(X.iloc[eval_index],
                           y.iloc[eval_index])
        # 依据验证集训练模型
        model_lgb = lgb.train(params, train_part, num_boost_round=5000,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=1000, verbose_eval=1000)
        # 测试集预测结果并纳入prediction_test容器
        model_list.append(model_lgb)
        train_pred += model_lgb.predict(X,num_iteration=model_lgb.best_iteration)
 
    return model_list,train_pred
