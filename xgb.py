def run_xgb(X, y):
    model_list = []
    params = {
          'eval_metric': 'rmse',
          'eta': 0.001,
          'max_depth': 10, 
          'subsample': 0.7, 
          'colsample_bytree': 0.7,
          'alpha':0.001,
          'random_state': 22}
    
    train_pred = np.zeros(len(X))
  # Part 4.交叉验证
    kf = KFold(n_splits=5, random_state=22, shuffle=True)
    n = 0
    print("training model...")
    for train_part_index, eval_index in kf.split(X, y):
        print("*"*20+" Kfold = {} ".format(n)+"*"*20)
        n = n + 1
        tr_data = xgb.DMatrix(X.iloc[train_part_index],y.iloc[train_part_index])
        va_data = xgb.DMatrix(X.iloc[eval_index],y.iloc[eval_index])
                              
        watchlist = [(tr_data, 'train'), (va_data, 'valid')]
        model_xgb = xgb.train(params, tr_data, 5000, watchlist, maximize=False, early_stopping_rounds = 1000, verbose_eval=1000)
        
        model_list.append(model_xgb)
        
        train_pred += model_xgb.predict(xgb.DMatrix(X),ntree_limit=model_xgb.best_ntree_limit)/5
        
    return  model_list,train_pred
