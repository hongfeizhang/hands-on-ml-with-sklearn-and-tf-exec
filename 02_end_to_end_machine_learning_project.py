import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import Imputer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import  LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.externals import joblib

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

HOUSING_PATH = "datasets/housing"

rooms_ix, bedrooms_ix, population_ix, household_ix =3,4,5,6


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def show_plot(plot):
    plot.hist(bins=50, figsize=(20, 15))
    plt.show()


def split_train_test(data, test_ratio):
    shuffled_data = np.random.permutation(data)
    test_set = shuffled_data[:int(test_ratio * len(shuffled_data))]
    train_set = shuffled_data[int(test_ratio * len(shuffled_data)):]
    return train_set, test_set


# 将数据根据收入中位数的分布  取样为训练集和测试集
def stratified_shuffle_split(data, n_splits=1, test_size=0.2, random_state=42):
    housing = data
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
    for set in (start_test_set, start_train_set):
        set.drop(["income_cat"], axis=1, inplace=True)
    return start_train_set, start_test_set


def data_clean(data):
    start_train_set = data
    housing = start_train_set


# 缺失值填充
def data_impute(data):
    housing = data
    imputer = Imputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    # print(imputer.statistics_)
    # print(housing_num.median().values)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    return X


# View the relevance of each attribute to the median price
def corr_matrix_special(data):
    housing = data
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    attributes = ["median_house_value", "median_income",
                  "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()


# Add extra attributes
def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]

    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]


# @pysnooper.snoop()
# Add extra attribute columns
def combine_attr_adder(housing):
    attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                     kw_args={"add_bedrooms_per_room": False})
    housing_extra_attribs = attr_adder.transform(housing.values)
    housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs,
        columns=list(housing.columns) + ["rooms_per_household",
                                         "population_per_household"])
    # print(housing_extra_attribs.head())
    return housing_extra_attribs


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


if __name__ == "__main__":
    # 读取原始数据
    housing = load_housing_data()
    # print(housing.head())
    # show_plot(housing)
    # train_set,test_set=split_train_test(housing,0.2)
    # train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)

    # 将数据根据收入中位数的分布  取样为训练集和测试集
    start_train_set, start_test_set = stratified_shuffle_split(housing)

    # housing = start_train_set
    # corr_matrix_special(housing)

    # 删除标签数据  将标签数单独保存
    housing = start_train_set.drop("median_house_value", axis=1)
    housing_labels = start_train_set["median_house_value"].copy()
    # 缺失值填充
    # data_impute(housing)

    # 取出数值数据
    housing_num = housing.drop('ocean_proximity', axis=1)
    # 取出非数值的分类数据
    housing_cat = housing[['ocean_proximity']]
    # print(housing_cat)

    # 将分类数据转换为独热向量
    # ordinal_encoder=OrdinalEncoder()
    # housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
    # print(housing_cat_encoded[:10])
    # print(ordinal_encoder.categories_)
    # cat_encoder = OneHotEncoder(sparse=False)
    # housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    # print(housing_cat_1hot)
    # print(housing_cat_1hot.toarray())
    # print(cat_encoder.categories_)
    # print(housing.columns)

    # rooms_ix, bedrooms_ix, population_ix, household_ix = [
    #   list(housing.columns).index(col)
    #  for col in ("total_rooms", "total_bedrooms", "population",
    #  "households")]

    # 添加额外的属性
    # combine_attr_adder(housing)

    # 流水线清理转换数据
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), #用中位数填缺失值
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)), #增加数据列
        ('std_scaler', StandardScaler()), #数据归归一化
    ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs), #处理文本和类型数据，将类型数据编码为独热向量
    ])

    housing_prepared=full_pipeline.fit_transform(housing)
    #print(housing_prepared)
    #print(housing_prepared.shape)

    '''
    lin_reg=LinearRegression()
    #lin_reg.fit(housing_prepared,housing_labels)

    #some_data = housing.iloc[:5]
    #some_labels = housing_labels.iloc[:5]
    #some_data_prepared = full_pipeline.transform(some_data)
    #print("Predictions:", lin_reg.predict(some_data_prepared))
    #print("Labels:",list(some_labels))
    #housing_predictions=lin_reg.predict(housing_prepared)
    #lin_mse=mean_squared_error(housing_labels,housing_predictions)
    #lin_rmse=np.sqrt(lin_mse)
    #print(lin_rmse)

    tree_reg=DecisionTreeRegressor(random_state=42)
    #tree_reg.fit(housing_prepared,housing_labels)

    #housing_predictions=tree_reg.predict(housing_prepared)
    #tree_mse=mean_squared_error(housing_labels,housing_predictions)
    #tree_rmse=np.sqrt(tree_mse)
    #print(tree_rmse)

    
    scores=cross_val_score(tree_reg,housing_prepared,housing_labels,
                           scoring="neg_mean_squared_error", cv=10)

    tree_rmse_scores=np.sqrt(-scores)
    display_scores(tree_rmse_scores)

    line_scores=cross_val_score(lin_reg,housing_prepared,housing_labels,
                                scoring="neg_mean_squared_error",cv=10)

    line_rmse_scores=np.sqrt(-line_scores)
    display_scores(line_rmse_scores)

    forest_reg=RandomForestRegressor(n_estimators=10,random_state=42)
    #print(forest_reg.fit(housing_prepared, housing_labels))

    #housing_predictions=forest_reg.predict(housing_prepared)
    #forest_mse = mean_squared_error(housing_labels, housing_predictions)
    #forest_rmse = np.sqrt(forest_mse)
    #print(forest_rmse)

    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)

    #保存模型
    #joblib.dump(my_model,"my_model.pkl")
    '''

    # param_grid告诉Scikit - Learn首先评估所有的列在第一个dict中
    # 的n_estimators和max_features的3×4 = 12种组合。
    # 然后尝试第二个dict中超参数的2×3 = 6种组合，这次会将超参
    # 数bootstrap设为False而不是True（后者是该超参数的默认值）。
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # 网格搜索会探索12+6=18种RandomForestRegressor的超参数组合，会训练每个模
    # 型五次（因为用的是五折交叉验证）。换句话说，训练总共有18×5=90轮！
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)
    print(grid_search.best_params_)
    pd.DataFrame(grid_search.cv_results_)

    '''
    #随机搜索
    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    '''

    feature_importances = grid_search.best_estimator_.feature_importances_
    feature_importances

    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    # cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    sorted(zip(feature_importances, attributes), reverse=True)

    final_model = grid_search.best_estimator_

    X_test = start_test_set.drop("median_house_value", axis=1)
    y_test = start_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)