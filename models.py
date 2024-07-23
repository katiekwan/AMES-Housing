# functions: load_ames(filename), data_encoding(X, y, log = 'no'), splitter(X, y, log = 'no'), linear(X, y, log = 'no', cont_trans = 'ss', model = 'linear', **kwargs)

import numpy as np
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# import os
# if not os.path.exists("images"): 
#     os.mkdir("images")
# pd.set_option('display.max_columns', None)
# from scipy.stats import pointbiserialr, f_oneway
# from scipy.stats import boxcoximport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

def load_ames():
    '''
    Takes a '.csv'. 
    Returns the loaded df as well as two global variables. 
    1. categorical - list of categorical features
    2. continuous - list of continuous features
    
    '''
    global categorical, continuous

    #load file
    housing_d = pd.read_csv('housing_cleaned.csv', index_col=0)
    housing_d = housing_d.convert_dtypes()
    housing_d = housing_d.reset_index().drop(columns = 'PID')
    
    data = pd.read_csv('30-year-fixed-mortgage-rate-chart.csv')
    data.date = pd.to_datetime(data['date'])
    data['year'] = data.date.dt.year
    data['month'] = data.date.dt.month
    housing_d = housing_d.merge(data, how = 'left', left_on = ('YrSold','MoSold'), right_on=('year','month')).drop(
    columns = ['year','month']).rename(columns = {' value':'rate'})
    

    #Feature engineer remodeled
    housing_d['remodeled'] = housing_d.YearBuilt - housing_d.YearRemodAdd
    housing_d['remodeled'] = housing_d['remodeled'].apply(lambda x: 1 if x < 0 else 0)
    
    continuous = ['MSSubClass','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
              'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
              'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
              'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',
              'Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
               'PoolQC','Fence','MiscFeature','SaleType','SaleCondition','rate','PoolArea',]
    
    categorical = ['SalePrice','GrLivArea','LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1',
              'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','BsmtFullBath',
             'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt',
              'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',
              'MiscVal','YrSold','OverallQual','OverallCond','MoSold','remodeled']
    return housing_d
    

def load_ames_nei():
    '''
    Takes a '.csv'. 
    Returns the loaded df as well as two global variables. 
    1. categorical - list of categorical features
    2. continuous - list of continuous features
    
    '''
    global categorical, continuous

    #load file
    housing_d = pd.read_csv('housing_cleaned.csv', index_col=0)
    housing_d = housing_d.convert_dtypes()
    housing_d = housing_d.reset_index().drop(columns = 'PID')
    
    data = pd.read_csv('30-year-fixed-mortgage-rate-chart.csv')
    data.date = pd.to_datetime(data['date'])
    data['year'] = data.date.dt.year
    data['month'] = data.date.dt.month
    housing_d = housing_d.merge(data, how = 'left', left_on = ('YrSold','MoSold'), right_on=('year','month')).drop(
    columns = ['year','month']).rename(columns = {' value':'rate'})

    #Feature engineer remodeled
    housing_d['remodeled'] = housing_d.YearBuilt - housing_d.YearRemodAdd
    housing_d['remodeled'] = housing_d['remodeled'].apply(lambda x: 1 if x < 0 else 0)
    

    #Add neighborhood quantiles
    nei_quant = pd.read_pickle('nei_quant.pkl')
    nei_quant.Feature = nei_quant.Feature.str.replace('Neighborhood_','')
    housing_d = housing_d.merge(nei_quant[['Feature','quant']], how = 'left', left_on = "Neighborhood", right_on = 'Feature').rename(
    columns = {'quant':'Nei_quant'})
    housing_d = housing_d.drop(columns = 'Feature')
    
    continuous = ['MSSubClass','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
              'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
              'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
              'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',
              'Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
               'PoolQC','Fence','MiscFeature','SaleType','SaleCondition','rate','PoolArea',]
    
    categorical = ['SalePrice','GrLivArea','LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1',
              'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','BsmtFullBath',
             'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt',
              'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',
              'MiscVal','YrSold','OverallQual','OverallCond','Nei_quant','remodeled','MoSold',]
    return housing_d

def load_cont():
    continuous = ['MSSubClass','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
              'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
              'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
              'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',
              'Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
               'PoolQC','Fence','MiscFeature','SaleType','SaleCondition','rate','PoolArea',]
    
    categorical = ['SalePrice','GrLivArea','LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1',
              'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','BsmtFullBath',
             'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt',
              'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',
              'MiscVal','YrSold','OverallQual','OverallCond','Nei_quant','remodeled','MoSold',]
    return continuous

def load_cat():
    continuous = ['MSSubClass','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
              'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
              'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
              'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',
              'Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
               'PoolQC','Fence','MiscFeature','SaleType','SaleCondition','rate']
    
    categorical = ['SalePrice','GrLivArea','LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1',
              'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','BsmtFullBath',
             'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt',
              'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
              'MiscVal','YrSold','OverallQual','OverallCond','Nei_quant','remodeled','MoSold',]
    return categorical
    
def data_encoding(X, y, log = 'no'):

    '''
    Applies One Hot Encoding
    
    X: independents
    y: dependent
    log: default 'no'. 'yes' for LogY or 'no' to pass. 
    '''
    import __main__
    
    #transform y
    if log == 'yes':
        y_mod = np.log(y)
    else:
        y_mod = y

    #define cat list for all categorical features
    cat = []
    for i in X.columns:
        if i in __main__.categorical:
            cat.append(i)

    #transform X
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_array = encoder.fit_transform(X[cat])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat))
    df_encoded = pd.concat([X, encoded_df], axis=1)
    X_mod = df_encoded.drop(cat, axis=1)

    return X_mod, y_mod, {'categoricals': cat}

def splitter(X, y, log = 'no'):

    '''
    one hot encodes and returns 80:20 train test split. X_train, X_test, y_train, y_test
    [0] = X_train, [1] = X_test, [2] = y_train, [3] = y_test, [4] cat list
    
    X: independents
    y: dependent
    log: default 'no'. 'yes' for LogY or 'no' to pass. 
    '''

    X_mod, y_mod, cat = data_encoding(X = X, y = y, log = log) #need to link return of function
        
    X_train, X_test, y_train, y_test = train_test_split(X_mod, y_mod, test_size=0.2, random_state=1)
    
    return X_train, X_test, y_train, y_test, cat
    

#Linear2.0 Standard Scaling the dummies

def linear(X, y, log = 'no', cont_trans = 'ss', model = 'linear', **kwargs):

    '''
    Returns a tuple with [0] = list of scores, [1] = dictionary of scores, [2] = features and coefficients,
    [3] = r2, [4] = adj r2, [5] = mse, [6] = abs error, [7] = rmse, [8] = intercept, [9] = standard list of metrics[10] = coefficients
    [11] = regressors
    
    X: independents
    y: dependent
    cat: default is []. Should be a list of all categorical features for One Hot Encoding
    log: default 'no'. 'yes' for LogY or 'no' to pass. 
    cont_trans: default is 'ss' = StandardScaler(). Can also pass 'n' = Normalizer().
    model: dedault is 'linear'. Can also pass 'lasso', 'ridge','elastic'.
    **kwargs will be passed to the models. 
    '''
    
    X_train, X_test, y_train, y_test, cat = splitter(X=X, y=y,log=log)

    num = list(X.columns)
    #choose con_trans
    if cont_trans == 'ss':
        transformer = StandardScaler()
        num = list(X_train.columns)
    elif cont_trans == 'n':
        transformer = Normalizer()
        #remove cat for Normalizer
        for i in cat['categoricals']:
            num.remove(i)
    else:
        raise ValueError("transformer must be either 'ss' or 'n'")
    

    #choose the model
    if model == 'linear':
        regressor = LinearRegression(**kwargs)
    elif model == 'lasso':
        regressor = Lasso(**kwargs)
    elif model == 'lassocv':
        regressor = LassoCV(**kwargs)
    elif model == 'ridge':
        regressor = Ridge(**kwargs)
    elif model == 'elastic':
        regressor = ElasticNet(**kwargs)
    else:
        raise ValueError("transforder must be either 'linear, lasso ridge, elastic'")
    
    
    
    # Define the column transformer 
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', transformer, num),
    ], remainder = 'passthrough')

    # Define the pipeline
    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
    ])

    #Fit the model
    pipeline.fit(X_train, y_train)

    #Predict the model
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    # Evaluate the model
    mse = round(mean_squared_error(y_test, y_pred),3)
    r2 = round(r2_score(y_test, y_pred),3)
    r2_train = round(r2_score(y_train, y_train_pred),3)
    
    rmse = round(np.sqrt(mse),3)
   
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis = 1)
        mae = round(np.mean(np.abs(y_pred - y_test)),3)
    else:
        mae = round(np.mean(np.abs(y_pred - y_test)),3)
        
    n = X_test.shape[0]
    p = X_test.shape[1] - 1  # Number of predictors
    adjusted_r2 = round((1 - (1 - r2) * (n - 1) / (n - p - 1)),3)

    #Extract feature names and coefficients
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    if model in ['linear','ridge']:
        coefficients = pipeline.named_steps['regressor'].coef_[0]
    else:
        coefficients = pipeline.named_steps['regressor'].coef_

    # Create a DataFrame to match feature names with coefficients
    coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
    }).sort_values(by = 'Coefficient', ascending = False)
    
    coef_df.Feature = coef_df.Feature.str.replace('num__','').str.replace('remainder__','')

    scores_dict = {'r2':r2, 'Adjusted r2':adjusted_r2, 'train r2': r2_train, 'MSE':mse, 'Abs Error':mae, 'RMSE': rmse}
    scores_list = [r2, adjusted_r2, r2_train, mse, mae, rmse]
    
    return scores_list, scores_dict, coef_df, r2, adjusted_r2, mse, mae, rmse, pipeline.named_steps['regressor'].intercept_,\
    ['r2','adj_r2','train_r2','mse','abs_err','rmse'], coefficients, pipeline.named_steps['regressor']

def encod_linear(df, ind, cont_trans = 'ss', model = 'linear', **kwargs):

    '''
    Returns a tuple with [0] = list of scores, [1] = dictionary of scores, [2] = features and coefficients,
    [3] = r2, [4] = adj r2, [5] = mse, [6] = abs error, [7] = rmse, [8] = intercept [9] = standrad list of scores
    [10] = coefficients, [11] = regressor
    
    df: dataframe inclusive of all encoded independents an dependents
    y: dependent as a string. 
    log: default 'no'. 'yes' for LogY or 'no' to pass. 
    cont_trans: default is 'ss' = StandardScaler(). Can also pass 'n' = Normalizer().
    model: dedault is 'linear'. Can also pass 'lasso', 'ridge','elastic'.
    **kwargs will be passed to the models. 
    '''

    X_mod = df.drop(columns = ind)
    y_mod = df[ind]
    num1 = list(X_mod.columns[~X_mod.columns.str.contains('_')])

    X_train, X_test, y_train, y_test = train_test_split(X_mod, y_mod, test_size=0.2, random_state=1)
    
    #choose con_trans
    if cont_trans == 'ss':
        transformer = StandardScaler()
    elif cont_trans == 'n':
        transformer = Normalizer()
    else:
        raise ValueError("transformer must be either 'ss' or 'n'")
    

    #choose the model
    if model == 'linear':
        regressor = LinearRegression(**kwargs)
    elif model == 'lasso':
        regressor = Lasso(**kwargs)
    elif model == 'lassocv':
        regressor = LassoCV(**kwargs)
    elif model == 'ridge':
        regressor = Ridge(**kwargs)
    elif model == 'elastic':
        regressor = ElasticNet(**kwargs)
    else:
        raise ValueError("transforder must be either 'linear, lasso, lassocv, ridge, elastic'")
    
    
    # Define the column transformer 
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', transformer, num1),
    ], remainder = 'passthrough')

    # Define the pipeline
    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
    ])

    #Fit the model
    pipeline.fit(X_train, y_train)

    #Predict the model
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    # Evaluate the model
    mse = round(mean_squared_error(y_test, y_pred),3)
    r2 = round(r2_score(y_test, y_pred),3)
    r2_train = round(r2_score(y_train, y_train_pred),3)
    
    rmse = round(np.sqrt(mse),3)
    
    mae='skip'
    # if y_pred.ndim == 1:
    #     y_pred = np.expand_dims(y_pred, axis = 1)
    #     mae = np.mean(np.abs(y_pred - y_test))
    # else:
    #     mae = np.mean(np.abs(y_pred - y_test))

    n = X_test.shape[0]
    p = X_test.shape[1] - 1  # Number of predictors
    adjusted_r2 = round((1 - (1 - r2) * (n - 1) / (n - p - 1)),3)

    #Extract feature names and coefficients
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    if model in ['linear','ridge']:
        coefficients = pipeline.named_steps['regressor'].coef_
    else:
        coefficients = pipeline.named_steps['regressor'].coef_[0]
    
    # Create a DataFrame to match feature names with coefficients
    coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
    }).sort_values(by = 'Coefficient', ascending = False)
    coef_df.Feature = coef_df.Feature.str.replace('num__','').str.replace('remainder__','')
    
    scores_dict = {'r2':r2, 'Adjusted r2':adjusted_r2, 'train r2': r2_train, 'MSE':mse, 'Abs Error':mae, 'RMSE': rmse}
    scores_list = [r2, adjusted_r2, r2_train, mse, mae, rmse]
    
    return scores_list, scores_dict, coef_df, r2, adjusted_r2, mse, mae, rmse, pipeline.named_steps['regressor'].intercept_,\
    ['r2','adj_r2','train_r2','mse','abs_err','rmse'], coefficients, pipeline.named_steps['regressor']

def split_linear(X_df, y_df, cont_trans = 'ss', model = 'linear', **kwargs):

    '''
    Returns a tuple with [0] = list of scores, [1] = dictionary of scores, [2] = features and coefficients,
    [3] = r2, [4] = adj r2, [5] = mse, [6] = abs error, [7] = rmse, [[8] = intercept, [9] = standrad list of scores,
    [10] = coefficients, [11] = regressor
    
    df: dataframe inclusive of all independents 
    y: series of dependents
    log: default 'no'. 'yes' for LogY or 'no' to pass. 
    cont_trans: default is 'ss' = StandardScaler(). Can also pass 'n' = Normalizer().
    model: dedault is 'linear'. Can also pass 'lasso', 'ridge','elastic'.
    **kwargs will be passed to the models. 
    '''

   
    num1 = list(X_df.columns[~X_df.columns.str.contains('_')])

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=1)
    
    #choose con_trans
    if cont_trans == 'ss':
        transformer = StandardScaler()
    elif cont_trans == 'n':
        transformer = Normalizer()
    else:
        raise ValueError("transformer must be either 'ss' or 'n'")
    

    #choose the model
    if model == 'linear':
        regressor = LinearRegression(**kwargs)
    elif model == 'lasso':
        regressor = Lasso(**kwargs)
    elif model == 'lassocv':
        regressor = LassoCV(**kwargs)
    elif model == 'ridge':
        regressor = Ridge(**kwargs)
    elif model == 'elastic':
        regressor = ElasticNet(**kwargs)
    else:
        raise ValueError("transforder must be either 'linear, lasso, lassocv, ridge, elastic'")
    
    
    # Define the column transformer 
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', transformer, num1),
    ], remainder = 'passthrough')

    # Define the pipeline
    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
    ])

    #Fit the model
    pipeline.fit(X_train, y_train)

    #Predict the model
    y_pred = pipeline.predict(X_test)
    # y_test = y_test.array
    y_train_pred = pipeline.predict(X_train)

    # Evaluate the model
    mse = round(mean_squared_error(y_test, y_pred),3)
    r2 = round(r2_score(y_test, y_pred),3)
    r2_train = round(r2_score(y_train, y_train_pred),3)
    rmse = round(np.sqrt(mse),3)
    mae='skip'
    
    # if y_pred.ndim == 1:
    #     y_pred = np.expand_dims(y_pred, axis = 1)
    #     mae = np.mean(np.abs(y_pred - y_test))
    # else:
    #     mae = np.mean(np.abs(y_pred - y_test))

    n = X_test.shape[0]
    p = X_test.shape[1] - 1  # Number of predictors
    adjusted_r2 = round((1 - (1 - r2) * (n - 1) / (n - p - 1)),3)

    #Extract feature names and coefficients
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    if model in ['linear','ridge']:
        coefficients = pipeline.named_steps['regressor'].coef_
    else:
        coefficients = pipeline.named_steps['regressor'].coef_
    
    # Create a DataFrame to match feature names with coefficients
    coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
    }).sort_values(by = 'Coefficient', ascending = False)
    coef_df.Feature = coef_df.Feature.str.replace('num__','').str.replace('remainder__','')
    
    scores_dict = {'r2':r2, 'Adjusted r2':adjusted_r2, 'train r2': r2_train, 'MSE':mse, 'Abs Error':mae, 'RMSE': rmse}
    scores_list = [r2, adjusted_r2, r2_train, mse, mae, rmse]
    
    return scores_list, scores_dict, coef_df, r2, adjusted_r2, mse, mae, rmse, pipeline.named_steps['regressor'].intercept_,\
    ['r2','adj_r2','train_r2','mse','abs_err','rmse'], coefficients, pipeline.named_steps['regressor']
    