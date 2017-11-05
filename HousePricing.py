import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib as plt
from scipy import stats
from sklearn.linear_model import LassoCV, LogisticRegressionCV, Ridge
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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


def analyse():

    train, test = load()
    all_data = pd.concat((train, test)).reset_index()  # type: pd.DataFrame
    all_data_orig = all_data.copy()

    # Add new features:
    # * Descriptive_GrLivArea
    all_data['Descriptive_GrLivArea'] = all_data['GrLivArea']\
        .map(lambda v: 'small' if v < 844 else 'medium' if v < 1761 else 'large')
    # Descriptive_TotalBsmtSF
    all_data['Descriptive_TotalBsmtSF'] = all_data['TotalBsmtSF']\
        .map(lambda v: 'small' if v < 960 else 'medium' if v < 1770 else 'large')

    # Check correlations
    # Credits to https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    corr_mat = all_data.corr()
    ax = sns.heatmap(corr_mat, square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Check correlations of top n features
    top_n = 10
    cols = corr_mat.nlargest(top_n, 'SalePrice')['SalePrice'].index
    ax = sns.heatmap(corr_mat.loc[cols, cols])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=22)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Multicollinearity present in
    # * CarageCars and CarageArea -> Choose CarageArea
    # * TotalBsmtSF and 1stFlrSF -> Choose 1stFlsSF
    multicoll_cols = ['GarageCars', 'TotalBsmtSF']
    cols = cols[~cols.isin(multicoll_cols)]
    all_data = all_data.loc[:, ~all_data.columns.isin(multicoll_cols)]

    # Check with scatterplots
    # * correlations
    # * outliers
    sns.pairplot(all_data[cols].dropna(), size=2.5)
    # Drop from train
    # * GrLivArea > 4000
    # * 1stFlrSF > 4000

    # Plot SalePrice histogram
    train.loc[:,'SalePrice'].plot(kind='hist')
    # Seems to be skewed

    # Check skewedness
    train.loc[:,'SalePrice'].skew()
    # Is skewed. Use log-transform for analysis.

    # Check other numeric variables too for skewedness.
    skewed_feats = get_skewed_feats(all_data)
    # Check the most significant skewed feats
    sign_skewed_feats = skewed_feats[skewed_feats.isin(cols)]
    ax = plt.pyplot.subplot()

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

    train = all_data.loc[slice(0,len(train)-1),:] # type: pd.DataFrame
    # Drop outliers
    train = train.drop(train[train['GrLivArea'] > 4000].index)
    train = train.drop(train[train['1stFlrSF'] > 4000].index)

    test = all_data.loc[slice(len(train), len(all_data)),:]
    test = test.loc[:, test.columns != 'SalePrice']

    X = train.loc[:,train.columns!='SalePrice']
    y = train.loc[:,'SalePrice']

    # Check what are the most significant parameters by coefficients
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    pipeline.fit(X, y)
    top_coefs = pd.DataFrame({'name': list(X), 'abs_coef': np.abs(pipeline.named_steps['ridge'].coef_)})\
        .sort_values('abs_coef')
    threshold = np.percentile(top_coefs['abs_coef'], 90)
    top_coefs = top_coefs.where(lambda r: r['abs_coef'] > threshold).dropna()
    ax = top_coefs.plot(kind='bar')
    ax.set_xticklabels(top_coefs['name'], rotation=45)

    model = Ridge()
    scores = cross_val_score(model, X, y, cv=KFold())

    model.fit(X, y)
    pred = model.predict(test)
    pred = np.expm1(pred)
    save(pd.DataFrame({'SalePrice': pred, 'Id': test['Id']}))

    pass


if __name__ == '__main__':
    random.seed(123)
    pd.set_option('display.width', 160)
    analyse()
