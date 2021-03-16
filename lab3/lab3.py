import numpy as np
import os
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class Num5Detector:

    number_detector = 5

    def __init__(self):
        np.random.seed(42)

        # To plot pretty figures
        mpl.rc('axes', labelsize=14)
        mpl.rc('xtick', labelsize=12)
        mpl.rc('ytick', labelsize=12)

        # Where to save the figures
        PROJECT_ROOT_DIR = "."
        IMAGE_DIR = "FIXME"


    def save_fig(self,fig_id, tight_layout=True):
        path = os.path.join(PROJECT_ROOT_DIR, "images", IMAGE_DIR, fig_id + ".png")
        if os.path.isfile(path):
            os.remove(path)   # Opt.: os.system("rm "+strFile)
        print("\n\nSaving figure...", fig_id)       
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format='png', dpi=300)        

    def random_digit(self):
        some_digit = X[36000]
        some_digit_image = some_digit.reshape(28, 28)
        plt.imshow(some_digit_image, cmap = mpl.cm.binary,
                interpolation="nearest")
        plt.axis("off")

        save_fig("fig_"+str(i))
        plt.show()


    # fetch_openml returns the unsorted MNIST dataset, whereas fetch_mldata() returned the dataset sorted by target (the training set and the test set were sorted separately). 
    def load_and_sort(self):
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml(name='mnist_784', version=1, cache=True)
            mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
            # sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
        except ImportError:
            from sklearn.datasets import fetch_mldata
            mnist = fetch_mldata('MNIST original')
            # mnist["data"], mnist["target"]
        return mnist

    def sort_by_target(self, mnist):
        train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
        test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
        X_train = mnist.data.iloc[train]
        Y_train = mnist.target.iloc[train]
        X_test = mnist.data.iloc[test + 60000]
        Y_test = mnist.target.iloc[test + 60000]
        return X_train, Y_train, X_test, Y_test

    def cal_cross_val(self,n, data):
        rmse_sum = 0
        for i in range(n):
            X_train, Y_train, X_test, Y_test = self.sort_by_target(data)
            sgd_clf = self.train_data(X_train, Y_train)
            Y_testpred = sgd_clf.predict(X_test)
            Y_testorig = (Y_test == self.number_detector)
            rmse_i = self.rmse(pd.DataFrame(Y_testorig.to_numpy()), pd.DataFrame(Y_testpred))
            print(f"rmse {i + 1}: {rmse_i}")
            rmse_sum += rmse_i
        return rmse_sum / n
 
    def train_pred(self,some_digit,X_train, Y_train):
        sgd_clf = train(X_train, Y_train)
        return sgd_clf.predict(pd.DataFrame(some_digit).transpose())

    def train_data(self,X_train, y_train):
        index = np.random.permutation(60000)
        X_train, Y_train = X_train.iloc[index], y_train.iloc[index]
        Y_train_clf = (Y_train == self.number_detector)
        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier()
        sgd_clf.fit(X_train, Y_train_clf)
        return sgd_clf

    def rmse(self,Y_actual, Y_hypothesis):
        return np.sqrt(mean_squared_error(Y_actual, Y_hypothesis))


def main():

    numobj= Num5Detector()

    mnist=numobj.load_and_sort()

    cross_val=numobj.cal_cross_val(5,mnist)

    print('Cross validation score : ', cross_val)

if __name__ == "__main__":
    main()

