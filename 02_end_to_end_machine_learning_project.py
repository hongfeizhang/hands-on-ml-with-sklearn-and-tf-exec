import sys
import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import  BaseEstimator,TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import pysnooper

from sklearn.preprocessing import Imputer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler

from sklearn.compose import ColumnTransformer

HOUSING_PATH = "datasets/housing"

rooms_ix, bedrooms_ix, population_ix, household_ix =3,4,5,6


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def show_plot(plot):
    plot.hist(bins=50, figsize=(20, 15))
    plt.show()

def split_train_test(data,test_ratio):
    shuffled_data=np.random.permutation(data)
    test_set=shuffled_data[:int(test_ratio*len(shuffled_data))]
    train_set=shuffled_data[int(test_ratio*len(shuffled_data)):]
    return train_set,test_set


#将数据根据收入中位数的分布  取样为训练集和测试集
def stratified_shuffle_split(data,n_splits=1,test_size=0.2,random_state=42):
    housing=data
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    # 将大于5的归入分类5
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        start_train_set = housing.loc[train_index]
        start_test_set = housing.loc[test_index]
    # print(housing["income_cat"].value_counts()/len(housing))
    # print(start_train_set["income_cat"].value_counts()/len(start_train_set))
    # print(start_test_set["income_cat"].value_counts()/len(start_test_set))
    for set in (start_test_set,start_train_set):
        set.drop(["income_cat"],axis=1,inplace=True)
    return start_train_set,start_test_set

def data_clean(data):
    start_train_set=data
    housing=start_train_set

#缺失值填充
def data_impute(data):
    housing=data
    imputer = Imputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    #print(imputer.statistics_)
    #print(housing_num.median().values)
    X=imputer.transform(housing_num)
    housing_tr=pd.DataFrame(X,columns=housing_num.columns)
    return  X

#查看每个属性与房价中位数的关联度
def corr_matrix_special(data):
    housing=data
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    attributes=["median_house_value","median_income",
                "total_rooms","housing_median_age"]
    scatter_matrix(housing[attributes] ,figsize=(12,8))
    plt.show()

#增加额外的属性
def add_extra_fetures(X,add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]

    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

#@pysnooper.snoop()
#增加额外的属性列
def combine_attr_adder(housing):
    attr_adder = FunctionTransformer(add_extra_fetures, validate=False,
                                     kw_args={"add_bedrooms_per_room": False})
    housing_extra_attribs = attr_adder.transform(housing.values)
    housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs,
        columns=list(housing.columns) + ["rooms_per_household",
                                         "population_per_household"])
    #print(housing_extra_attribs.head())
    return housing_extra_attribs

def num_pipeline(housing_num):
    num_pipeline=Pipeline([
        ('imputer',Imputer(strategy="median")),
        ('attribs_adder',FunctionTransformer(add_extra_fetures,validate=False)),
        ('std_scaler',StandardScaler()),
    ])
    housing_num_tr=num_pipeline.fit_transform(housing_num)
    return housing_num_tr

def column_transform(housing_num):
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared

if __name__ == "__main__":
    #读取原始数据
    housing = load_housing_data()
    #print(housing.head())
    #show_plot(housing)
    #train_set,test_set=split_train_test(housing,0.2)
    #train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)

    #将数据根据收入中位数的分布  取样为训练集和测试集
    start_train_set,start_test_set=stratified_shuffle_split(housing)

    #housing = start_train_set
    # corr_matrix_special(housing)

    #删除标签数据  将标签数单独保存
    housing=start_train_set.drop("median_house_value",axis=1)
    housing_labels=start_train_set["median_house_value"].copy()

    # 缺失值填充
    #data_impute(housing)


    housing_num = housing.drop('ocean_proximity', axis=1)
    #取出非数值的分类数据
    housing_cat = housing[['ocean_proximity']]
    #print(housing_cat)


    #将分类数据转换为独热向量
    ordinal_encoder=OrdinalEncoder()
    housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
    #print(housing_cat_encoded[:10])
    #print(ordinal_encoder.categories_)
    cat_encoder = OneHotEncoder(sparse=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    #print(housing_cat_1hot)
    #print(housing_cat_1hot.toarray())
    #print(cat_encoder.categories_)
    #print(housing.columns)

    #rooms_ix, bedrooms_ix, population_ix, household_ix = [
    #   list(housing.columns).index(col)
    #  for col in ("total_rooms", "total_bedrooms", "population",
    #  "households")]

    #添加额外的属性
    combine_attr_adder(housing)

    #流水线转换
    housing_num_tr=num_pipeline(housing_num)
    print(housing_num_tr)











