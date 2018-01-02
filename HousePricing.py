import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LassoLars, LinearRegression
from sklearn.preprocessing.data import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import random
import xgboost as xgb

def rmsle(real,predicted,saleprice_is_skewed):
    sum=0.0
    for x in range(len(predicted)):
        if saleprice_is_skewed:
            p = np.array(predicted)[x]
            r = np.array(real)[x]
        else:
            p = np.log(np.array(predicted)[x]+1)
            r = np.log(np.array(real)[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


def load() -> (pd.DataFrame, pd.DataFrame):
    train_data = pd.read_csv('data\\train.csv')
    test_data = pd.read_csv('data\\test.csv')
    return train_data, test_data


def save(predicted_data: pd.DataFrame):
    saved_data = predicted_data.loc[:,['Id','SalePrice']] # type: pd.DataFrame
    saved_data.to_csv('data\\predicted.csv', index=False)


def get_grade_map(addition: dict = None):
    grade_map = {np.nan:0,'Po':1,'Fa':2,'Ta':3,'TA':3,'Av':3,'Gd':4,'Ex':5}
    if addition is not None:
        grade_map = {**grade_map, **addition}
    return grade_map


def get_xgb():
    return xgb.XGBRegressor(
        colsample_bytree=0.2,
        gamma=0.0,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=1.5,
        n_estimators=7200,
        reg_alpha=0.3,
        reg_lambda=0.3,
        subsample=0.2,
        seed=42,
        silent=1
    )


def evaluate(X, y, sale_price_is_skewed):
    models = [('Lasso', Lasso()),
              ('Ridge', Ridge()),
              ('LnReg', LinearRegression()),
              ('XGBoo', get_xgb())]

    scorer = make_scorer(rmsle, greater_is_better=False, saleprice_is_skewed=sale_price_is_skewed)

    for n, m in models:
        cv_scores = cross_val_score(m, X, y, scoring=scorer)
        print(' ' + n + ' rmsle %0.5f (+/-%0.5f)' % (cv_scores.mean(), cv_scores.std()*2))


def analyse():

    # Load data
    train, test = load()
    full = pd.concat((train, test)).reset_index(drop=True) # type: pd.DataFrame
    train_ix = list(range(0, len(train)))
    test_ix = list(range(len(train), len(full)))

    # Obtain features
    feat_sale_price = 'SalePrice'

    # Create a stupid model with nothing but
    # * NA's filled with 0 for int64, float64 (except for SalePrice) and skewness fixed
    # * dummies generated for categoricals such that NA is its own
    # and see scores.
    full_stupid = full.copy(deep=True)

    def guess_missing_with_just_something_not_completely_unreasonable(data_w_missing):

        feats_numeric = pd.Series(data_w_missing.select_dtypes(['int32','int64', 'float64']).columns.values)
        feats_numeric_no_price = feats_numeric[feats_numeric != feat_sale_price]
        feats_text = data_w_missing.columns[~data_w_missing.columns.isin(feats_numeric)]

        data_w_missing.loc[:, feats_numeric_no_price] = data_w_missing[feats_numeric_no_price].fillna(0)
        data_w_missing = pd.get_dummies(data_w_missing, columns=feats_text, dummy_na=True)
        return data_w_missing

    def fix_skewness(data_w_skewed):

        feats_numeric = pd.Series(data_w_skewed.select_dtypes(['int64', 'float64']).columns.values)

        # TODO: do some hyper-parameter search for skewness_threshold
        skewness_threshold = .2
        # TODO: is it enough to fix actual numeric feats, or should some dummies be included
        feats_skewed = data_w_skewed[feats_numeric].skew().abs()\
            .where(lambda v: v > skewness_threshold).dropna().index.values
        data_w_skewed.loc[:, feats_skewed] = np.log1p(data_w_skewed[feats_skewed])
        return data_w_skewed, feat_sale_price in feats_skewed

    def slice_x_and_y(data_w_x_and_y, ix_for_slice):
        sliced_x = data_w_x_and_y.loc[ix_for_slice, :].drop(feat_sale_price, 1)
        sliced_y = data_w_x_and_y.loc[ix_for_slice, feat_sale_price]
        return sliced_x, sliced_y

    def do_high_level_feature_engineering(data_to_engineer):

        # "Do whatever makes the evaluation step not crash"
        data_to_engineer = guess_missing_with_just_something_not_completely_unreasonable(data_to_engineer)
        data_to_engineer, sale_price_is_skewed = fix_skewness(data_to_engineer)

        # Using the stupid model...
        # Scores with the setup above:
        # Lasso rmsle -0.39491 (+/-0.01787)
        # Ridge rmsle -0.13036 (+/-0.01788)
        # LnReg rmsle -0.14094 (+/-0.02656)
        # XGBoo rmsle -0.12477 (+/-0.01799)

        # Not too stupid. But maybe it is still possible to add some mass transformation
        # and thus increase the score?

        # Normalization does not seem to make the score any higher

        # Adding new scaled features for all numerics makes Lasso better, but not better than what Ridge gives
        # without adding the new feats,

        return data_to_engineer, sale_price_is_skewed

    full_stupid, sale_price_is_skewed = do_high_level_feature_engineering(full_stupid)
    stupid_x, stupid_y = slice_x_and_y(full_stupid, train_ix)

    print("Stupid model scores:")
    evaluate(stupid_x, stupid_y, sale_price_is_skewed)

    # Do low level feature engineering

    # Check out nulls
    # null_cols = pd.isnull(full).sum() # type: pd.Series
    # null_cols.sort_values(ascending=False, inplace=True)
    # null_cols = null_cols.drop('SalePrice')
    # null_cols = pd.DataFrame({'nullAmt': null_cols}, index=null_cols.index)
    # print('Amount of nulls per column and distinct values for categoricals:')
    # null_cols['type'] = full.loc[:,null_cols.index].dtypes
    # null_cols['dist'] = full.loc[:,null_cols.index[null_cols['type']=='object']]\
    #     .apply(lambda c: ','.join(pd.Series(c).dropna().unique()))
    # print(null_cols)

    # * Feature engineering does not seem to have effect

    full, sale_price_is_skewed = do_high_level_feature_engineering(full)

    # Score
    print("Engineered model scores:")
    evaluate(full.loc[train_ix,:].drop('SalePrice',1), full.loc[train_ix, 'SalePrice'], sale_price_is_skewed)

    # Save the scores:
    model = get_xgb()
    model.fit(full.loc[train_ix,:].drop('SalePrice',1), full.loc[train_ix, 'SalePrice'])
    full.loc[test_ix, 'SalePrice'] = model.predict(full.loc[test_ix, :].drop('SalePrice',1))

    # Revert the log transformation
    if sale_price_is_skewed:
        print("Fix skewed sale price")
        full.loc[:, 'SalePrice'] = np.expm1(full['SalePrice'])

    save(full.loc[test_ix, :])

    pass


if __name__ == '__main__':
    random.seed(123)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', 100)
    analyse()
