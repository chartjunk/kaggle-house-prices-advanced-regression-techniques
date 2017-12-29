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


def evaluate(X, y, sale_price_is_skewed):
    models = [('Lasso', Lasso()),
              ('Ridge', Ridge()),
              ('LnReg', LinearRegression()),
              ('XGBoo', xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1))]

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
    feats_numeric = pd.Series(full.select_dtypes(['int64', 'float64']).columns.values)
    feats_numeric_no_price = feats_numeric[feats_numeric != feat_sale_price]
    feats_text = full.columns[~full.columns.isin(feats_numeric)]

    def guess_missing_with_just_something_not_completely_unreasonable(data_w_missing):
        data_w_missing.loc[:, feats_numeric_no_price] = data_w_missing[feats_numeric_no_price].fillna(0)
        data_w_missing = pd.get_dummies(data_w_missing, columns=feats_text, dummy_na=True)
        return data_w_missing

    def fix_skewness(data_w_skewed):
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
    evaluate(stupid_x, stupid_y, sale_price_is_skewed)

    # Save the scores:
    model = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
    model.fit(full_stupid.loc[train_ix, :].drop('SalePrice', 1), full_stupid.loc[train_ix, 'SalePrice'])
    full_stupid.loc[test_ix, 'SalePrice'] = model.predict(full_stupid.loc[test_ix, :].drop('SalePrice', 1))

    # Revert the log transformation
    if sale_price_is_skewed:
        full_stupid.loc[:, 'SalePrice'] = np.expm1(full_stupid['SalePrice'])

    save(full_stupid.loc[test_ix, :])

    # TODO: apply feature engineering

    return

    # Check out nulls
    null_cols = pd.isnull(full).sum() # type: pd.Series
    null_cols.sort_values(ascending=False, inplace=True)
    null_cols = null_cols.drop('SalePrice')
    null_cols = pd.DataFrame({'nullAmt': null_cols}, index=null_cols.index)
    print('Amount of nulls per column and distinct values for categoricals:')
    null_cols['type'] = full.loc[:,null_cols.index].dtypes
    null_cols['dist'] = full.loc[:,null_cols.index[null_cols['type']=='object']]\
        .apply(lambda c: ','.join(pd.Series(c).dropna().unique()))

    print(null_cols)

    # Cv accuracy without any of the following engineering:
    # Lasso accuracy 0.81143 (+/-0.13061)
    # Ridge accuracy 0.83060 (+/-0.10870)
    # LnReg accuracy 0.77877 (+/-0.13678)

    # So let's engineer some features
    # > PoolQC
    # * Lots of nulls. Possibly means that there's no pool
    # * Map into ordered value
    full.loc[:, 'PoolQC'] = full.loc[:, 'PoolQC']\
        .map(get_grade_map())\
        .astype('int64')

    # > MiscFeature
    # * Nulls possibly mean that there is no miscellaneous features
    # * Othr value seems shady. Drop it.
    # * Binarise and drop the column
    full.loc[:, 'MiscFeature_Shed'] = pd.Series(full.loc[:, 'MiscFeature']=='Shed')*1
    full.loc[:, 'MiscFeature_Gar2'] = pd.Series(full.loc[:, 'MiscFeature']=='Gar2')*1
    full.loc[:, 'MiscFeature_TenC'] = pd.Series(full.loc[:, 'MiscFeature']=='TenC')*1
    full.drop('MiscFeature',1,inplace=True)

    # > Alley
    # * Lots of nulls. Possibly means that there's no alley
    # * Also generally Grvl seems worse than Pave
    full.loc[:, 'Alley'] = full.loc[:, 'Alley']\
        .map({np.nan: 0, 'Grvl': 2, 'Pave':3})\
        .astype('int64')

    # > Fence
    # * The following procedure deceases accuracy, so skip it
    #full.loc[:, 'Fence_Prv'] = pd.Series(full.loc[:, 'Fence'])\
    #    .map({'MnPrv':1, 'GdPrv':2}).fillna(0).astype('int64')
    #full.loc[:, 'Fence_Wd'] = pd.Series(full.loc[:, 'Fence'])\
    #    .map({'MnWv':1, 'GdWo':2}).fillna(0).astype('int64')

    # > FireplaceQu
    full.loc[:, 'FireplaceQu'] = full.loc[:, 'FireplaceQu']\
        .map(get_grade_map()).astype('int64')

    # > LotFrontage
    # * The following procedure decreases accuracy, so skip it
    #full.loc[:, 'HasFrontage'] = ~(full.loc[:, 'LotFrontage'].isnull())*1
    #full.loc[:, 'LotFrontage'].fillna(0, inplace=True)
    #full.loc[:, 'LotFrontage'] = full.loc[:, 'LotFrontage'].astype('int64')

    # > GarageFinish
    # * Null probably means that there's no garage
    #   * All Garage* columns have equal amount of nulls except for GarageType
    # * Make ordered, because finished is better than unfinished
    full.loc[:, 'HasGarage'] = ~(full.loc[:, 'GarageFinish'].isnull())*1
    full.loc[:, 'GarageFinish'] = full.loc[:, 'GarageFinish']\
        .map({np.nan: 0, 'Unf': 2, 'RFn': 3, 'Fin': 4})\
        .astype('int64')

    # > GarageCond
    full.loc[:, 'GarageCond'] = full.loc[:, 'GarageCond']\
        .map(get_grade_map()).astype('int64')

    # > GarageQual
    full.loc[:, 'GarageQual'] = full.loc[:, 'GarageQual']\
        .map(get_grade_map()).astype('int64')

    # > GarageYrBlt
    # * Min value is 1895, so let's assume that having no garage is worse than having
    #   a garage more ancient than that
    full.loc[:, 'GarageYrBlt'] = full.loc[:, 'GarageYrBlt'].fillna(1800).astype('int64')

    # > GarageType
    # * The following procedure decreases the accuracy, so skip it
    #full.loc[full.loc[:, 'HasGarage'] == 0, 'GarageType'] = np.nan
    #full.loc[:, 'GarageType'].fillna('NoGarage', inplace=True)
    # * Create new features, because has mixed semantics
    #full['GarageIsAttached'] = (full['GarageType'].isin(['Attchd','BuiltIn']))*1
    #full['GarageInBasement'] = (full['GarageType'] == 'Basment')*1
    #full['HasCarPort'] = (full['GarageType'] == 'CarPort')*1
    #full.drop('GarageType', axis=1, inplace=True)
    # NOTE: could be valuable information to extract the distance from garage
    # * Attchd = 0, BuiltIn = 0, Detchd = 1, Basment = 2, CarPort = 2

    # > Bsmt*
    # * Doing the following procedure to the basement features decreases the score, so skip it
    #bsmt_cols = ['BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']
    # * Bsmt columns have almost equal amount of nulls
    #   * Let's see if BsmtFinType1 can be used to determine whether there's a
    #     basement or not
    full.loc[:, 'HasBasement'] = ~(full.loc[:, 'BsmtFinType1'].isnull())*1
    #print(full.loc[full.loc[:, 'HasBasement'] == 1, bsmt_cols]\
    #    .isnull().sum())

    # Yes, replace the leftover nulls with an "unknown" value
    #full.loc[full.loc[:, 'HasBasement'] == 0, bsmt_cols] = 'Unknown'

    # > BsmtCond
    # * Replace Unknown with the same score as Fa
    #full.loc[:, 'BsmtCond'] = full.loc[:, 'BsmtCond']\
    #    .map(get_grade_map({'Unknown':2})).astype('int64')

    # > BsmtExposure
    #full.loc[:, 'BsmtExposure'] = full.loc[:, 'BsmtExposure']\
    #    .map({np.nan:0,'No':1,'Mn':2,'Av':3,'Gd':4,'Unknown':2}) \
    #    .astype('int64')

    # > BsmtQual
    #full.loc[:, 'BsmtQual'] = full.loc[:, 'BsmtQual']\
    #    .map(get_grade_map({'Unknown':2})).astype('int64')

    # > BsmtFinType1 and 2
    # * Columns are semantically the same
    # * Turn into new columns:
    #   * BsmtLivingQuartersQual
    #   * BsmtHasRecRoom
    bsmt_fin_type_glq_3 = np.any(full.loc[full.loc[:, 'HasBasement'] == 1, ['BsmtFinType1', 'BsmtFinType2']] == 'GLQ', 1)
    bsmt_fin_type_alq_2 = np.any(full.loc[full.loc[:, 'HasBasement'] == 1, ['BsmtFinType1', 'BsmtFinType2']] == 'ALQ', 1)
    bsmt_fin_type_blq_1 = np.any(full.loc[full.loc[:, 'HasBasement'] == 1, ['BsmtFinType1', 'BsmtFinType2']] == 'BLQ', 1)
    bsmt_fin_type_rec_1 = np.any(full.loc[full.loc[:, 'HasBasement'] == 1, ['BsmtFinType1', 'BsmtFinType2']] == 'Rec', 1)
    bsmt_fin_type_lwq_0 = np.any(full.loc[full.loc[:, 'HasBasement'] == 1, ['BsmtFinType1', 'BsmtFinType2']] == 'LwQ', 1)
    bsmt_fin_type_unf_0 = np.any(full.loc[full.loc[:, 'HasBasement'] == 1, ['BsmtFinType1', 'BsmtFinType2']] == 'Unf', 1)
    # All of these have a substantial amount of True-values, so let's make use of all of them
    full.loc[bsmt_fin_type_glq_3, 'BsmtLivingQuartersQual'] = 3
    full.loc[bsmt_fin_type_alq_2, 'BsmtLivingQuartersQual'] = 2
    full.loc[bsmt_fin_type_blq_1, 'BsmtLivingQuartersQual'] = 1
    full.loc[bsmt_fin_type_lwq_0, 'BsmtLivingQuartersQual'] = 0
    full.loc[bsmt_fin_type_unf_0, 'BsmtLivingQuartersQual'] = 0
    full.loc[:, 'BsmtLivingQuartersQual'] = full.loc[:, 'BsmtLivingQuartersQual'].fillna(0).astype('int64')
    full.loc[bsmt_fin_type_rec_1, 'HasRecRoom'] = 1
    full.loc[:, 'HasRecRoom'] = full.loc[:, 'HasRecRoom'].fillna(0).astype('int64')
    full.drop('BsmtFinType1', 1, inplace=True)
    full.drop('BsmtFinType2', 1, inplace=True)

    # > MasVnrType and MasVnrArea
    # * Both have almost equal amount of nulls, so probably means that there's no masonry veneer
    full.loc[:, 'MasVnrType'] = full.loc[:, 'MasVnrType'].fillna('None')
    full.loc[:, 'MasVnrArea'] = full.loc[:, 'MasVnrArea'].fillna(0).astype('int64')
    full.loc[:, 'HasMasVnr'] = (full.loc[:, 'MasVnrType'] != 'None')*1

    # > MSZoning
    # * Handle nulls by replacing with the mode
    print('MSZoning value_counts:')
    print(full.loc[:, 'MSZoning'].value_counts())
    # > RL is a clear winner
    #full.loc[:, 'MSZoning'].fillna('RL', inplace=True)

    # * Transform into two new columns:
    #   * IsCommZone
    #   * IsResidZone
    #   * ResidDensity
    full.loc[:, 'IsCommZone'] = (full.loc[:, 'MSZoning'] == 'C (all)')*1
    full.loc[:, 'IsResidZone'] = (full.loc[:, 'MSZoning'].isin(['RL', 'RM', 'FV', 'RH']))*1
    full.loc[:, 'ResidDensity'] = full.loc[:, 'MSZoning'].map({'RL':1,'RM':2,'RH':3,'FV':2})
    full.loc[:, 'ResidDensity'].fillna(0, inplace=True)
    full.loc[:, 'ResidDensity'] = full.loc[:, 'ResidDensity'].astype('int64')
    full.drop('MSZoning', 1, inplace=True)

    # > BsmtFullBath and BsmtHalfBath
    full.loc[:, ['BsmtFullBath','BsmtHalfBath']].fillna(0, inplace=True)
    full.loc[:, 'BsmtBath'] = full.loc[:, 'BsmtFullBath'] + .5*full.loc[:, 'BsmtHalfBath']

    # > Utilities
    # * Only one with NoSeWa. Keeping this column seems to increase the cv accuracy, though, so let it be

    # > Functional
    # * Handle nulls by replacing with the mode
    print('Functional value_counts:')
    print(full.loc[:,'Functional'].value_counts())
    # > Typ is the most popular hands down
    full.loc[:, 'Functional'].fillna('Typ', inplace=True)
    full.loc[:, 'Functional'] = full.loc[:, 'Functional']\
        .map({'Typ':0,'Min1':1,'Min2':2,'Mod':3,'Maj1':4,'Maj2':5,'Sev':6,'Sal':7}).astype('int64')
    full.loc[:, 'HasFunctionalDeductions'] = (full.loc[:, 'Functional'] != 0)*1

    # > Electrical
    # * Does not seem to effect the result. Drop
    full.drop('Electrical', 1, inplace=True)

    # > Exterior1st and 2nd
    full['Exterior1st'].fillna('VinylSd', inplace=True)
    full['Exterior2nd'].fillna('VinylSd', inplace=True)
    exterior_values = np.union1d(
        full['Exterior1st'].value_counts().index,
        full['Exterior2nd'].value_counts().index)
    for v in exterior_values:
        full.loc[:,'Exterior_' + v] = (pd.concat([full['Exterior1st'] == v, full['Exterior2nd'] == v], axis=1)\
            .any(axis=1))*1
    full.drop('Exterior1st', 1, inplace=True)
    full.drop('Exterior2nd', 1, inplace=True)

    # > KitchenQual. does not seem to improve score.
    # > GarageCars
    full['IsZeroCars'] = (full['GarageCars'] == 0)*1
    # * Assumption: The first car adds more to the price than the following cars
    full['GarageCarsLog'] = np.log1p(full['GarageCars'])

    # > TotalBsmtSF


    #...

    # > ExterQual
    full.loc[:, 'ExterQual'] = full['ExterQual'].map(get_grade_map()).astype('int64')

    # > ExterCond
    full.loc[:, 'ExterCond'] = full['ExterCond'].map(get_grade_map()).astype('int64')

    # > HeatingQC
    full.loc[:, 'HeatingQC'] = full['HeatingQC'].map(get_grade_map()).astype('int64')

    # TODO: Rest of the feature engineering stuff

    # Create dummies for the rest of the categorical values
    full = pd.get_dummies(full, dummy_na=True)

    # Replace rest of numeric missing values with median
    numeric_feats = full.select_dtypes(['int32','int64','float64']).columns.values
    numeric_feats_missing = full[numeric_feats].isnull().sum().drop('SalePrice')\
        .where(lambda v: v > 0).dropna().index.values
    full.fillna(full[numeric_feats_missing].median(), inplace=True)

    # Fix skewed features
    skewed_feats = full.skew().abs().where(lambda v: v > 0.22).dropna().index.values
    full.loc[:, skewed_feats] = np.log1p(full[skewed_feats])

    saleprice_is_skewed = np.any(skewed_feats == 'SalePrice')

    # Score
    evaluate(full.loc[train_ix,:].drop('SalePrice',1), full.loc[train_ix, 'SalePrice'], saleprice_is_skewed)

    # Save the scores:
    model = Ridge()
    model.fit(full.loc[train_ix,:].drop('SalePrice',1), full.loc[train_ix, 'SalePrice'])
    full.loc[test_ix, 'SalePrice'] = model.predict(full.loc[test_ix, :].drop('SalePrice',1))

    # Revert the log transformation
    if saleprice_is_skewed:
        full.loc[:, 'SalePrice'] = np.expm1(full['SalePrice'])

    save(full.loc[test_ix, :])

    pass


if __name__ == '__main__':
    random.seed(123)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', 100)
    analyse()
