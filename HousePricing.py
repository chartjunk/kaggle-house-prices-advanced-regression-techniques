import numpy as np
import pandas as pd
import random
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


def rmsle(predicted,real):
    sum=0.0
    for x in range(len(predicted)):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


def load() -> (pd.DataFrame, pd.DataFrame):
    train_data = pd.read_csv('data\\train.csv')
    test_data = pd.read_csv('data\\test.csv')
    return train_data, test_data


def save(predicted_data: pd.DataFrame):
    saved_data = predicted_data.loc[:,['Id','SalePrice']] # type: pd.DataFrame
    saved_data.to_csv('data\\predicted.csv', index=False)


def get_numeric_feats(data: pd.DataFrame) -> pd.Series:
    numeric_feats = data.dtypes.apply(lambda v: v in ['int64', 'float64'])
    numeric_feats = data.dtypes[numeric_feats].index
    return numeric_feats


def get_skewed_feats(data: pd.DataFrame) -> pd.Series:
    max_skew = 0.7
    numeric_feats = get_numeric_feats(data)
    skewed_feats = data[numeric_feats].skew().where(lambda v: v > max_skew).dropna().index
    return skewed_feats


def get_missing_feats(data: pd.DataFrame) -> pd.Series:
    missing_feats = np.sum(pd.isnull(data)).where(lambda v: v > 100).dropna().index
    missing_feats = missing_feats[missing_feats != 'SalePrice']
    return missing_feats


def explore():

    train, test = load()

    # Plot SalePrice histogram
    train.loc[:,'SalePrice'].plot(kind='hist')
    # Seems to be skewed

    # Check skewedness
    train.loc[:,'SalePrice'].skew()
    # Is skewed. Use log-transform for analysis.

    # Check other numeric variables too for skewedness.
    all_data = pd.concat((train, test)).reset_index() # type: pd.DataFrame
    skewed_feats = get_skewed_feats(all_data)

    # Take GrLivArea as an example of a skewed feature.
    all_data.loc[:,'GrLivArea'].plot(kind='hist')

    # Fix skewed features.
    all_data.loc[:,skewed_feats] = np.log1p(all_data.loc[:,skewed_feats])

    # Check GrLivArea again for normality
    all_data.loc[:, 'GrLivArea'].plot(kind='hist')
    # Normality seems to be much better now.

    # Generate dummies for categoricals
    all_data = pd.get_dummies(all_data)

    # Check missing value amounts
    missing_feats = get_missing_feats(all_data)
    # Drop features with too many missing values
    all_data = all_data.drop(missing_feats, axis=1) # type: pd.DataFrame
    # Still some missing values. Impute with median.
    all_data = all_data.fillna(all_data.median())

    train = all_data.loc[slice(0,len(train)-1),:]
    test = all_data.loc[slice(len(train), len(all_data)),:]

    X = train.loc[:,train.columns!='SalePrice'] # type: pd.DataFrame
    y = train.loc[:,'SalePrice']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    # Define hyperparameter space for ridge. Try different alphas.
    parameters = {'ridge__alpha': np.logspace(-4, 0)}
    cv = GridSearchCV(pipeline, parameters, cv=5)
    cv.fit(X, y)

    # Check consistency with cross validation
    scores1 = cross_val_score(cv, X, y, cv=KFold())
    # Ridge seems pretty good

    pass


def analyse():

    train, test = load()
    all_data = pd.concat((train, test)).reset_index()
    skewed_feats = get_skewed_feats(all_data)
    all_data.loc[:, skewed_feats] = np.log1p(all_data.loc[:, skewed_feats])
    all_data = pd.get_dummies(all_data)
    missing_feats = get_missing_feats(all_data)
    all_data = all_data.drop(missing_feats, axis=1)  # type: pd.DataFrame
    all_data = all_data.fillna(all_data.median())
    train = all_data.loc[slice(0, len(train) - 1), :]
    test = all_data.loc[slice(len(train), len(all_data)), :]
    X = train.loc[:,train.columns!='SalePrice']
    y = train.loc[:,'SalePrice']
    test = test.loc[:,test.columns!='SalePrice']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    # Define hyperparameter space for ridge. Try different alphas.
    parameters = {'ridge__alpha': np.logspace(-4, 0)}
    model = GridSearchCV(pipeline, parameters, cv=5)

    model.fit(X, y)
    pred = model.predict(test)
    pred = np.expm1(pred)
    save(pd.DataFrame({'SalePrice':pred,'Id':test['Id']}))
    pass


if __name__ == '__main__':
    random.seed(123)
    pd.set_option('display.width', 160)
    explore()
    #analyse()
