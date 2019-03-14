from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import  clone

if __name__ = "__main__":

    mnist = fetch_mldata('MNIST original', data_home='./')
    #print (mnist)

    X,y=mnist["data"],mnist["target"]
    print(X.shape)
    print(y.shape)

    some_digit=X[36000]
    some_digit_image=some_digit.reshape(28,28)
    plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
    plt.axis("off")
    #plt.show()

    print (y[36000])
    X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]
    shuffle_index=np.random.permutation(60000)
    X_train,y_train=X_train[shuffle_index],y_train[shuffle_index]

    y_train_5=(y_train==5)
    y_test_5=(y_test==5)

    from sklearn.linear_model import SGDClassifier
    sgd_clf=SGDClassifier(random_state=42)
    sgd_clf.fit(X_train,y_train_5)

    print(sgd_clf.predict([some_digit]))

    skfolds
