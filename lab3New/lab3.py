import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_selection import RFE


        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        # print(self.pima.head())
        self.X_test = None
        self.y_test = None
    

    # def define_feature(self):
    #     X  = self.pima.iloc[:, :-1]
    #     y =  self.pima.iloc[:,-1]
    #     plt.figure(figsize=(12,10))
    #     cor = self.pima.corr()
    #     sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
    #     # plt.show()
    #     cor_target = abs(cor["label"])
    #     relevant_features = cor_target[cor_target>0.2]
    #     print("features == ", relevant_features)
        # feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        # X = self.pima[feature_cols]
        # y = self.pima.label
        # return X, y
    
    def train(self,x):
        # split X and y into training and testing sets
        X, y = self.feature_def(x)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y,random_state=12345)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(solver='lbfgs', max_iter=10000)
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self,x):
        model = self.train(x)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        # print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)


    def set_bp(self):
        newBp = pd.Series(
            ["low", "high"], dtype="category")
        self.pima["newBp"] = newBp

        self.pima.loc[self.pima["bp"] < 50, "newBp"] = newBp[0]
        self.pima.loc[self.pima["bp"] > 50, "newBp"] = newBp[1]

    def set_glucose(self):
        newGlucose = pd.Series(
            ["normal", "prediabetes", "diabetic"], dtype="category")
        self.pima["newGlucose"] = newGlucose

        self.pima.loc[self.pima["glucose"] < 140, "newGlucose"] = newGlucose[0]
        self.pima.loc[((self.pima["glucose"] > 140) & (
            self.pima["glucose"] <= 200)), "newGlucose"] = newGlucose[1]
        self.pima.loc[self.pima["glucose"] > 200, "newGlucose"] = newGlucose[2]

    def set_insulin(self, row):
        if row["insulin"] >= 16 and row["insulin"] <= 166:
            return "Normal"
        else:
            return "Abnormal"

    def feature_select(self):
        # Creating new logical feature variables from the existing features for better prediction values
        # we can create two new features new_age and new_bmi
        # self.pima = self.pima.assign(
        #     newGlucose=self.pima.apply(self.set_glucose, axis=1))
        # self.pima = self.pima.assign(
        #     newBmi=self.pima.apply(self.set_bmi, axis=1))
        self.set_glucose()
        self.set_bp()
        self.pima = self.pima.assign(
            newInsulin=self.pima.apply(self.set_insulin, axis=1))
        self.pima = pd.get_dummies(
            self.pima, columns=["newGlucose", "newInsulin","newBp"])
        # print(self.pima.head())

    def feature_def(self,x):
        model = LogisticRegression(solver='lbfgs', max_iter=10000,random_state=12345)

        self.pima = self.pima[['newGlucose_normal', 'newGlucose_prediabetes','newInsulin_Normal','newInsulin_Abnormal', 'newBp_low','newBp_high',
                                 'pregnant', 'pedigree', 'insulin', 'skin', 'bp', 'age', 'label']]
        # print("pima", self.pima)
        array = self.pima.values
        # print("array = ", array)

        X = array[:, 0:12]
        y = array[:, 12]
        rfe = RFE(model, x)
        fit = rfe.fit(X, y)
        # print("Num Features: %d" % fit.n_features_)
        # print("Selected Features: %s" % fit.support_)
        # print("Feature Ranking: %s" % fit.ranking_)
        reduced_dataset = self.pima.iloc[:, :-1].loc[:, fit.support_]
        # print(reduced_dataset.head())
        return reduced_dataset, self.pima.label
    
if __name__ == "__main__":
    classifer = DiabetesClassifier()
    
    classifer.feature_select()
    for x in [6,7,8] :
        classifer.feature_def(x)
        result = classifer.predict(x)
        # print(f"Predicition={result}")
        score = classifer.calculate_accuracy(result)
        print(f"score={score}")
        con_matrix = classifer.confusion_matrix(result)
        print(f"confusion_matrix=${con_matrix}")




