import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer


def rmsle(predicted,real):
    sum=0.0
    for x in range(len(predicted)):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


def load():
    train_data = pd.read_csv('data\\train.csv')
    test_data = pd.read_csv('data\\test.csv')
    return train_data, test_data


def save(predicted_data: pd.DataFrame):
    saved_data = predicted_data.loc[:,['Id','SalePrice']] # type: pd.DataFrame
    saved_data.to_csv('data\\predicted.csv', index=False)


def to_cat(data: pd.DataFrame, name: str, ordered_cats=None):
    if ordered_cats is not None:
        data[name] = data[name].astype(str).astype('category', categories=ordered_cats, ordered=True)
    else:
        data[name] = data[name].astype(str).astype('category')


def to_no_cat(data: pd.DataFrame, name: str, ordered_cats=None):
    data.loc[:, name].fillna('No', inplace=True)
    to_cat(data, name, ordered_cats)


def impute_med(data: pd.DataFrame, name: str):
    data.loc[:, name].fillna(data.loc[:,name].median(), inplace=True)


def impute_0(data: pd.DataFrame, name: str):
    data.loc[:, name].fillna(0, inplace=True)


def impute_mfc(data: pd.DataFrame, name: str, ordered_cats=None):
    data.loc[:, name].fillna(data.loc[:, name].mode().iloc[0], inplace=True)
    to_no_cat(data, name, ordered_cats)


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    # Explore null columns:
    smry = pd.DataFrame({'dtype': data.dtypes,
                         'distAmt': data.nunique(),
                         'nullAmt': data.isnull().sum()})

    #                distAmt    dtype  nullAmt
    # 1stFlrSF          1083    int64        0
    # 2ndFlrSF           635    int64        0
    # 3SsnPorch           31    int64        0
    # Alley                2   object     2721
    # BedroomAbvGr         8    int64        0
    # BldgType             5   object        0
    # BsmtCond             4   object       82
    # BsmtExposure         4   object       82
    # BsmtFinSF1         991  float64        1
    # BsmtFinSF2         272  float64        1
    # BsmtFinType1         6   object       79
    # BsmtFinType2         6   object       80
    # BsmtFullBath         4  float64        2
    # BsmtHalfBath         3  float64        2
    # BsmtQual             4   object       81
    # BsmtUnfSF         1135  float64        1
    # CentralAir           2   object        0
    # Condition1           9   object        0
    # Condition2           8   object        0
    # Electrical           5   object        1
    # EnclosedPorch      183    int64        0
    # ExterCond            5   object        0
    # ExterQual            4   object        0
    # Exterior1st         15   object        1
    # Exterior2nd         16   object        1
    # Fence                4   object     2348
    # FireplaceQu          5   object     1420
    # Fireplaces           5    int64        0
    # Foundation           6   object        0
    # FullBath             5    int64        0
    # Functional           7   object        2
    # GarageArea         603  float64        1
    # GarageCars           6  float64        1
    # GarageCond           5   object      159
    # GarageFinish         3   object      159
    # GarageQual           5   object      159
    # GarageType           6   object      157
    # GarageYrBlt        103  float64      159
    # GrLivArea         1292    int64        0
    # HalfBath             3    int64        0
    # Heating              6   object        0
    # HeatingQC            5   object        0
    # HouseStyle           8   object        0
    # Id                2919    int64        0
    # KitchenAbvGr         4    int64        0
    # KitchenQual          4   object        1
    # LandContour          4   object        0
    # LandSlope            3   object        0
    # LotArea           1951    int64        0
    # LotConfig            5   object        0
    # LotFrontage        128  float64      486
    # LotShape             4   object        0
    # LowQualFinSF        36    int64        0
    # MSSubClass          16    int64        0
    # MSZoning             5   object        4
    # MasVnrArea         444  float64       23
    # MasVnrType           4   object       24
    # MiscFeature          4   object     2814
    # MiscVal             38    int64        0
    # MoSold              12    int64        0
    # Neighborhood        25   object        0
    # OpenPorchSF        252    int64        0
    # OverallCond          9    int64        0
    # OverallQual         10    int64        0
    # PavedDrive           3   object        0
    # PoolArea            14    int64        0
    # PoolQC               3   object     2909
    # RoofMatl             8   object        0
    # RoofStyle            6   object        0
    # SaleCondition        6   object        0
    # SalePrice          663  float64     1459
    # SaleType             9   object        1
    # ScreenPorch        121    int64        0
    # Street               2   object        0
    # TotRmsAbvGrd        14    int64        0
    # TotalBsmtSF       1058  float64        1
    # Utilities            2   object        2
    # WoodDeckSF         379    int64        0
    # YearBuilt          118    int64        0
    # YearRemodAdd        61    int64        0
    # YrSold               5    int64        0

    quality_levels = ['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex']

    # 1stFlrSF OK
    # 2ndFlrSF OK
    # 3SsnPorch OK
    to_no_cat(data, 'Alley') # Note: Could this be ordered: No<Grvl<Pave ?
    # BedroomAbvGr OK
    to_cat(data, 'BldgType')
    to_no_cat(data, 'BsmtCond', quality_levels)
    to_no_cat(data, 'BsmtExposure')
    # ATM impute low priority Bsmt missing values with median
    # * Is there a relationship between missing values and whether there is a basement or not
    impute_med(data, 'BsmtFinSF1')
    impute_med(data, 'BsmtFinSF2')
    to_no_cat(data, 'BsmtFinType1', ['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])
    to_no_cat(data, 'BsmtFinType2', ['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])
    impute_med(data, 'BsmtFullBath')
    impute_med(data, 'BsmtHalfBath')
    to_no_cat(data, 'BsmtQual', quality_levels)
    impute_med(data, 'BsmtUnfSF')
    to_cat(data, 'CentralAir')
    to_cat(data, 'Condition1')
    to_cat(data, 'Condition2')
    impute_mfc(data, 'Electrical')
    # EnclosedPorch OK
    to_cat(data, 'ExterCond', quality_levels)
    to_cat(data, 'ExterQual', quality_levels)
    impute_mfc(data, 'Exterior1st')
    impute_mfc(data, 'Exterior2nd')
    # Fence has 5 levels:
    #   GdPrv Good Privacy
    #   MnPrv Minimum Privacy
    #   GdWo  Good Wood
    #   MnWw  Minimum Wood/Wire
    #   NA    No Fence
    # o It may me worthwhile to turn this into an ordered category
    #   > Prioritising Good > Minimum, assume Wood < Privacy
    # TODO: separate into two columns instead
    to_no_cat(data, 'Fence', ['No', 'MnWw', 'MnPrv', 'GdWo', 'GdPrv'])
    to_no_cat(data, 'FireplaceQu', quality_levels)
    # Fireplaces OK
    to_cat(data, 'Foundation')
    # FullBath OK
    impute_mfc(data, 'Functional', ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'])
    impute_med(data, 'GarageArea')
    impute_med(data, 'GarageCars')
    to_no_cat(data, 'GarageCond', quality_levels)
    to_no_cat(data, 'GarageFinish')
    to_no_cat(data, 'GarageQual', quality_levels)
    to_no_cat(data, 'GarageType')
    # GarageYrBlt has an outlier, so clip:
    # >>> data.loc[:,'GarageYrBlt'].plot(kind='hist',logy=True)
    data.loc[:,'GarageYrBlt'].where(lambda v: v <= 2017, other=2017, inplace=True)
    impute_0(data, 'GarageYrBlt') # TODO: find out better neutral correspondence than 0
    # GrLivArea OK
    # HalfBath OK
    to_cat(data, 'Heating')
    to_cat(data, 'HeatingQC', quality_levels)
    to_cat(data, 'HouseStyle')  # Note: Could this be ordered?
    # Id OK
    # KitchenAbvGr OK
    to_no_cat(data, 'KitchenQual', quality_levels)
    to_cat(data, 'LandContour')
    to_cat(data, 'LandSlope', ['Gtl','Mod','Sev'])
    # LotArea OK
    to_cat(data, 'LotConfig')
    impute_0(data, 'LotFrontage')  # Note: smallest value is 22.0, so maybe imputing with 0 is justifiable
    to_cat(data, 'LotShape', ['Reg','IR1','IR2','IR3'])
    # LowQualFinSF OK
    to_cat(data, 'MSSubClass')
    # For MSZoning
    # >>> sns.countplot(x='MSSubClass', hue='MSZoning', data=full_data)
    # reveals that there may be some correlation between MSSubClass and MSZoning.
    # * fill missing values with argmax
    data.loc[data['MSSubClass'].isin([30,70]) & data['MSZoning'].isnull(), 'MSZoning'] = 'RM'
    data.loc[(data['MSSubClass']==20) & data['MSZoning'].isnull(), 'MSZoning'] = 'RL'
    to_cat(data, 'MSZoning')
    impute_0(data, 'MasVnrArea')
    data.loc[:, 'MasVnrType'].replace('None', 'No', inplace=True)
    to_no_cat(data, 'MasVnrType')
    to_no_cat(data, 'MiscFeature')  # TODO: MiscFeature + MiscVal into own boolean columns
    # MiscVal OK
    to_cat(data, 'MoSold')
    to_cat(data, 'Neighborhood')
    # OpenPorchSF OK
    # OverallCond OK
    # OverallQual OK
    to_cat(data, 'PavedDrive')
    # PoolArea OK
    to_no_cat(data, 'PoolQC', quality_levels)
    to_cat(data, 'RoofMatl')
    to_cat(data, 'RoofStyle')
    to_cat(data, 'SaleCondition')
    # SalePrice OK
    # For SaleType
    # >>> sns.countplot(x='SaleCondition', hue='SaleType', data=full_data)
    # reveals that there may be some correlation between SaleCondition and SaleType
    # * fill missing value with argmax
    data.loc[:,'SaleType'].fillna('WD', inplace=True)
    to_cat(data, 'SaleType')
    # ScreenPorch OK
    to_cat(data, 'Street')
    # TotRmsAbvGrd OK
    # For TotalBsmtSF, the missing value belongs to a row without any other Bsmt-data either
    impute_0(data, 'TotalBsmtSF')
    # For Utilities, most rows have value AllPub:
    # >>> full_data.loc[:,'Utilities'].value_counts()
    # so conform:
    impute_mfc(data, 'Utilities')
    # WoodDeckSF OK
    # YearBuilt OK
    # YearRemodAdd OK
    # YrSold OK

    # Check that there's no missing values left
    assert(not (data.loc['train', :].isnull().any().any()))
    assert(not (data.loc['test', :].drop('SalePrice', axis=1, ).isnull().any().any()))

    return data


def vectorize_ordered(data: pd.DataFrame, name: str):
    data[name] = pd.Categorical.from_array(data[name]).codes


def vectorize_unordered(data: pd.DataFrame, name: str):
    b = LabelBinarizer()
    new_data = b.fit_transform(data[name])
    if len(b.classes_) > 2:
        data.drop(name, 1, inplace=True)
        ix = 0
        for c in b.classes_:
            data[name + '__' + c] = new_data[:,ix]
            ix += 1
    else:
        data[name] = new_data


def vectorize(data: pd.DataFrame) -> pd.DataFrame:
    vectorize_unordered(data, 'Alley')
    vectorize_unordered(data, 'BldgType')
    vectorize_ordered(data, 'BsmtCond')
    vectorize_unordered(data, 'BsmtExposure')
    vectorize_ordered(data, 'BsmtFinType1')
    vectorize_ordered(data, 'BsmtFinType2')
    vectorize_ordered(data, 'BsmtQual')
    vectorize_unordered(data, 'CentralAir')
    vectorize_unordered(data, 'Condition1')
    vectorize_unordered(data, 'Condition2')
    vectorize_unordered(data, 'Electrical')
    vectorize_ordered(data, 'ExterCond')
    vectorize_ordered(data, 'ExterQual')
    vectorize_unordered(data, 'Exterior1st')
    vectorize_unordered(data, 'Exterior2nd')
    vectorize_ordered(data, 'Fence')
    vectorize_ordered(data, 'FireplaceQu')
    vectorize_unordered(data, 'Foundation')
    vectorize_ordered(data, 'Functional')
    vectorize_ordered(data, 'GarageCond')
    vectorize_unordered(data, 'GarageFinish')
    vectorize_ordered(data, 'GarageQual')
    vectorize_unordered(data, 'GarageType')
    vectorize_unordered(data, 'Heating')
    vectorize_ordered(data, 'HeatingQC')
    vectorize_unordered(data, 'HouseStyle')
    vectorize_ordered(data, 'KitchenQual')
    vectorize_unordered(data, 'LandContour')
    vectorize_ordered(data, 'LandSlope')
    vectorize_unordered(data, 'LotConfig')
    vectorize_ordered(data, 'LotShape')
    vectorize_unordered(data, 'MSSubClass')
    vectorize_unordered(data, 'MSZoning')
    vectorize_unordered(data, 'MasVnrType')
    vectorize_unordered(data, 'MiscFeature')
    vectorize_unordered(data, 'MoSold')
    vectorize_unordered(data, 'Neighborhood')
    vectorize_unordered(data, 'PavedDrive')
    vectorize_ordered(data, 'PoolQC')
    vectorize_unordered(data, 'RoofMatl')
    vectorize_unordered(data, 'RoofStyle')
    vectorize_unordered(data, 'SaleCondition')
    vectorize_unordered(data, 'SaleType')
    vectorize_unordered(data, 'Street')
    vectorize_unordered(data, 'Utilities')
    return data


def get_model(x: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    clf = RandomForestClassifier()
    clf.fit(x, y)
    return clf


def get_data() -> (pd.DataFrame, pd.DataFrame):
    train, test = load()
    full = pd.concat({'train': train, 'test': test})
    full = vectorize(normalize(full))  # TODO: additional features?
    train = full.loc['train']
    test = full.loc['test'].drop('SalePrice', 1)
    return train, test


def separate_xy(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    return data.drop('SalePrice',1), data['SalePrice']


def validate():

    test_size = 0.2

    train_full, _ = get_data()
    x_train_full, y_train_full = separate_xy(train_full)
    x_train, x_valid, y_train, y_valid = \
        train_test_split(x_train_full, y_train_full, test_size=test_size)
    model = get_model(x_train, y_train)
    y_valid_pred = model.predict(x_valid)
    score = mean_squared_error(y_valid, y_valid_pred)**.5
    print('Score = ' + str(score))


def predict():
    train, x_test = get_data()
    x_train, y_train = separate_xy(train)
    model = get_model(x_train, y_train)
    y_test_pred = model.predict(x_test)
    test_pred = pd.DataFrame({'Id':x_test['Id'], 'SalePrice':y_test_pred})
    save(test_pred)

if __name__ == '__main__':
    random.seed(123)
    pd.set_option('display.width', 160)
    validate()
    # predict()
