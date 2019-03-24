import sys
import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

HOUSING_PATH = "datasets/housing"


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
    return start_train_set,start_test_set

if __name__ == "__main__":
    housing = load_housing_data()
    #print(housing.head())
    #show_plot(housing)
    #train_set,test_set=split_train_test(housing,0.2)
    #train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)

    start_train_set,start_test_set=stratified_shuffle_split(housing)

    for set in (start_test_set,start_train_set):
        set.drop(["income_cat"],axis=1,inplace=True)

    housing=start_train_set
    corr_matrix=housing.corr()
    #for



