from sklearn import pipeline
from sklearn.metrics import fbeta_score, make_scorer, classification_report
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def BatchWalkForwardCV(df, 
                       n_splits, 
                       max_train_size, 
                       features, 
                       target, 
                       scaler,
                       model,
                       param_grid,
                       verbose=True):
    """ADD DOCSTRING
    """
    n_splits = n_splits
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
    all_dates = df.index.get_level_values(0).unique()

    preds = []
    symbols = []
    oos_dates = []
    test_score = []
    training_score = []
    for index_train, index_test in tscv.split(all_dates):
        
        X = df[features]
        y = df[target]

        #Establish training & test dates
        train_date_range = X.index.get_level_values(0).unique()[index_train]
        test_date_range = X.index.get_level_values(0).unique()[index_test]
        
        #------------------------------------------------------------------#
        
        #Select training and test data for features & targets
        X_train = X[X.index.get_level_values(0).isin(train_date_range)]
        X_test = X[X.index.get_level_values(0).isin(test_date_range)]
        
        y_train = y[y.index.get_level_values(0).isin(train_date_range)]
        y_test = y[y.index.get_level_values(0).isin(test_date_range)]
        
        #------------------------------------------------------------------#

        #Define stetps
        steps = [('scaler', scaler),
                 ('clf', model)]
        
        #Create a pipeline
        pipe = pipeline.Pipeline(steps=steps)
        
        #------------------------------------------------------------------#
        
        #Create scoring function
        fscore = make_scorer(fbeta_score, beta=.5, average='macro')

        #Rename param_grid
        params = {"clf__" +k:v for (k,v) in param_grid.items()}
        
        #Fit data and make predictions with tscv
        iterations = 10
        model_searchcv = RandomizedSearchCV(pipe, 
                                            params, 
                                            n_iter=iterations,
                                            scoring=fscore, 
                                            refit=fscore,
                                            random_state=5
                                            )
        
        model_searchcv.fit(X_train, y_train.values.ravel())  
        
        #------------------------------------------------------------------#
        
        #Store Results
        y_pred = model_searchcv.predict(X_test)
        preds.append(y_pred)
        test_dates = X_test.index.get_level_values(0) 
        oos_dates.append(test_dates)
        symbols.append(X_test.index.get_level_values(1))
        
        test_scores = fbeta_score(y_test, y_pred, beta=.5) #Testing f-score
        train_scores = model_searchcv.best_score_
        test_score.append(test_scores)
        training_score.append(train_scores)
        
        #------------------------------------------------------------------#
        if verbose == True:
            #Print Results
            print("Model:\n {}\n".format(model_searchcv.best_estimator_))
            print("#=================#")
            print("Best CV Score: {:.2f}".format(train_scores)) #Best cross-val score - training
            print("Test Score: {:.2f}".format(test_scores)) #Score from test set
            print("#=================#")
            print("Best Params: {}".format(model_searchcv.best_params_))
            print("Scorer: {}".format(model_searchcv.scorer_))
            
            print("\nClassification Report:", 
                  "\nFrom: {} - To: {}\n".format(test_dates.unique()[0], test_dates.unique()[-1]), 
                  "\n",
                  classification_report(
                      y_test,
                      y_pred,
                      target_names=['Underperform', 'Outperform']
                  )
                 )        
            
            print("#============================================#")
            print("#============================================#\n")

    return preds, symbols, oos_dates, test_score, training_score