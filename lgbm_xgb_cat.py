def stacking_models(X, y,test_X,test_Y):
    lgb_params = {'bagging_fraction': 0.9,'objective':'regression', 'metric':'rmse',"verbosity" : -1,'bagging_seed':42,
                'bagging_freq': 10,  'feature_fraction': 0.8,  'learning_rate': 0.01,  'max_depth': 5,  'min_data_in_leaf': 8, 
                'num_leaves': 30,  'reg_alpha': 31, 'reg_lambda': 40}
    xgb_params = {'eval_metric': 'rmse', 'eta': 0.001,  'max_depth': 10,  'subsample': 0.7, 
                  'colsample_bytree': 0.7, 'alpha':0.001,'random_state': 22}
        
  # Part 4.交叉验证
    kf = KFold(n_splits=5, random_state=52, shuffle=True)
    n = 0
    print("training model...")
    lgbm_train_pred = pd.DataFrame(data = np.zeros((len(X),2)),columns=["lgbm_train_pred","train_y"])
    xgb_train_pred = pd.DataFrame(data = np.zeros((len(X),2)),columns=["xgb_train_pred","train_y"])
    cat_train_pred = pd.DataFrame(data = np.zeros((len(X),2)),columns=["cat_train_pred","train_y"])
    
    lgbm_test_pred, xgb_test_pred, cat_test_pred = np.zeros(len(test_X)),np.zeros(len(test_X)),np.zeros(len(test_X))
    
    lgbm_model_list, xgb_model_list, cat_model_list = [],[],[]
    
    
    for train_part_index, eval_index in kf.split(X, y):
        print("*"*20+" Kfold = {} ".format(n)+"*"*20)
        n = n + 1
        train_X,train_y = X.iloc[train_part_index],y.iloc[train_part_index]
        val_X,val_y = X.iloc[eval_index],y.iloc[eval_index]
        
        ################### lgbm ###################
        print("*"*15+" Kfold = {}  training lgbm ".format(n)+"*"*15)
        train_part = lgb.Dataset(X.iloc[train_part_index],y.iloc[train_part_index])
        eval = lgb.Dataset(X.iloc[eval_index],y.iloc[eval_index])
        model_lgb = lgb.train(lgb_params, train_part, num_boost_round=5000,valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],early_stopping_rounds=1000, verbose_eval=1000)
        
        # prediction train
        train_pred =  model_lgb.predict(val_X,num_iteration=model_lgb.best_iteration)
        lgbm_train_pred.loc[eval_index,"lgbm_train_pred"] = train_pred
        lgbm_train_pred.loc[eval_index,"train_y"] = val_y
        
        # test_prediction
        test_pred =  model_lgb.predict(test_X,num_iteration=model_lgb.best_iteration)
        lgbm_test_pred += test_pred/5
        # save lgbm model 
        lgbm_model_list.append(model_lgb)
        
        
        ################### xgboost ###################
        print("*"*15+" Kfold = {}  training xgboost ".format(n)+"*"*15)
        tr_data = xgb.DMatrix(X.iloc[train_part_index],y.iloc[train_part_index])
        va_data = xgb.DMatrix(X.iloc[eval_index],y.iloc[eval_index])
        watchlist = [(tr_data, 'train'), (va_data, 'valid')]
        model_xgb = xgb.train(xgb_params, tr_data, 5000, watchlist, maximize=False, early_stopping_rounds = 1000, verbose_eval=1000)
        
        # catboost train prediction
        train_pred = model_xgb.predict(va_data,ntree_limit=model_xgb.best_ntree_limit)
        xgb_train_pred.loc[eval_index,"xgb_train_pred"] = train_pred
        xgb_train_pred.loc[eval_index,"train_y"] = val_y
   
        # xgboost test prediction
        test_pred =  model_xgb.predict(xgb.DMatrix(test_X),ntree_limit=model_xgb.best_ntree_limit)
        xgb_test_pred += test_pred/5
        
        # save xgboost model 
        xgb_model_list.append(model_xgb)
        
        
        ################### catboost ###################
        print("*"*15+" Kfold = {}  training catboost ".format(n)+"*"*15)
        model_cat = CatBoostRegressor(iterations=5000,learning_rate=0.05,depth=5,
                             eval_metric='RMSE',random_seed = 42,bagging_temperature = 1,od_type='Iter',od_wait=1000)
        model_cat.fit(train_X, train_y,eval_set=(val_X, val_y),use_best_model=True,verbose=1000)
        
        # catboost train prediction
        train_pred = model_cat.predict(val_X)
        cat_train_pred.loc[eval_index,"cat_train_pred"] = train_pred
        cat_train_pred.loc[eval_index,"train_y"] = val_y
   
        # catboost test prediction
        test_pred =  model_cat.predict(test_X)
        cat_test_pred += test_pred/5
        
        # save xgboost model 
        cat_model_list.append(model_cat)
    
    ################### agg train output ###################
    train_pred = pd.DataFrame()
    train_pred = pd.concat([train_pred,lgbm_train_pred],axis = 0)
    train_pred = pd.concat([train_pred,xgb_train_pred],axis = 0)
    train_pred = pd.concat([train_pred,cat_train_pred],axis = 0)
    
    ################### agg test output ###################
    test_pred = pd.DataFrame()
    lgbm_test_pred = pd.DataFrame(lgbm_test_pred,columns=["lgbm_test_pred"])
    xgb_test_pred = pd.DataFrame(xgb_test_pred,columns=["xgb_test_pred"])
    cat_test_pred = pd.DataFrame(cat_test_pred,columns=["cat_test_pred"])
    
    test_pred = pd.concat([test_pred,lgbm_test_pred],axis = 0)
    test_pred = pd.concat([test_pred,xgb_test_pred],axis = 0)
    test_pred = pd.concat([test_pred,cat_test_pred],axis = 0)
    test_pred["test_Y"] = test_Y
    
    return lgbm_model_list, xgb_model_list, cat_model_list,train_pred,test_pred
