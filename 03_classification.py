from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import  clone
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")

def ShowDigitPicture(some_digit):
    some_digit_image=some_digit.reshape(28,28)
    plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
    plt.axis("off")
    plt.show()

def Cross_Val_Score(X_train,y_train,Classifier):
    skfolds=StratifiedKFold(n_splits=3,random_state=42)
    for train_index,test_index in skfolds.split(X_train,y_train):
        clone_clf=clone(sgd_clf)
        X_train_folds=X_train[train_index]
        y_train_folds=(y_train[train_index])
        X_test_fold=X_train[test_index]
        y_test_fold=(y_train[test_index])
        clone_clf.fit(X_train_folds,y_train_folds)
        y_pred=clone_clf.predict(X_test_fold)
        n_correct=sum(y_pred==y_test_fold)
        print(n_correct/len(y_pred))


if __name__ == "__main__":

    mnist = fetch_mldata('MNIST original', data_home='./datasets/')
    #print (mnist)

    X,y=mnist["data"],mnist["target"]
    #print(X.shape)
    #print(y.shape)

    some_digit=X[36000]
    #ShowDigitPicture(some_digit)

    #print (y[36000])
    X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]
    shuffle_index=np.random.permutation(60000)
    X_train,y_train=X_train[shuffle_index],y_train[shuffle_index]

    print(X_train)
    y_train_5=(y_train==5)
    print(y_train_5)
    y_test_5=(y_test==5)
    print(y_test_5)



    sgd_clf=SGDClassifier(random_state=42)
    #print(type(sgd_clf))
    sgd_clf.fit(X_train,y_train_5)

    print(sgd_clf.predict([some_digit]))
    Cross_Val_Score(X_train,y_train_5,sgd_clf)


