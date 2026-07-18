import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#DATA PROCCESING
df = pd.read_csv("cardio_train.csv", sep=";")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

#LOAD DATASET
#missing values
print("\nMissing values: ")
print(df.isnull().sum())

#duplicate rows
print("\nduplicate rows: ")
print(df.duplicated().sum())

#data types
print("\ndata types: ")
print(df.dtypes)

#dataset
print("\ndataset shape: ")
print(df.shape)

#HEART DISEASE DISTRUBUTION
import matplotlib.pyplot as plt

df['cardio'].value_counts().plot(kind='bar')
plt.title("Heart disease distrubution")
plt.xlabel("cardio (0 = NO disease, 1 = Disease)")
plt.ylabel("count")
plt.show()

#GENDER DISTRUBUTION
plt.figure(figsize=(6,4))
sns.countplot(x = "gender", data = df )
plt.title("gender distrubution")
plt.show()
print("Graphs Completed")


#AGE DISTRUBUTION
plt.figure(figsize=(8,5))
plt.hist(df["age"]/365, bins = 30)
plt.title("age distrubution")
plt.xlabel(("age (years)"))
plt.ylabel("count")
plt.show()
print("Graphs Completed")

#CORRELATION MATRIX
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("correlatio matrix")
plt.show()
print("Graphs Completed")

#MECHINE LEARNING
x = df.drop("cardio", axis = 1)
y = df["cardio"]
print("features shape", x.shape)
print("target shape", y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42

)
print("training data: ", X_train.shape)
print("testing data: ", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#create model
log_model = LogisticRegression(max_iter = 1000)

#train model
log_model.fit(X_train, y_train)

#predict
y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#accuracy
print("\n===== logistic Regression=====")
print("accuracy: ", accuracy)

#CONFUSION MATRIX
print("\nconfusion matrix: ")
print(confusion_matrix(y_test, y_pred))

#CLASSIFICATION REPORT
print("\nclassification report: ")
print(classification_report(y_test, y_pred))


 #FEATURES SCALING
 
# FEATURE SCALING
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Completed")

#KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("\n===== knn classifier=====")
print("accuracy: ", knn_accuracy)

print("\nconfusion matrix: ")
print(confusion_matrix(y_test, y_pred_knn))

print("\nclassification report: ")
print(classification_report(y_test, y_pred_knn))



#DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier


dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\n=====decision tree classifier=====")
print("accuracy", dt_accuracy)

print("\n=====confusion matrix=====")
print(confusion_matrix( y_test, y_pred_dt))

print("\n=====classification report=====")
print(classification_report( y_test, y_pred_dt))

#RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier(
    n_estimators=30,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\n=====randaom forest=====")
print("accuracy", rf_accuracy)

print("\nconfusion matrix=====")
print(confusion_matrix(y_test, y_pred_rf))

print("\nclassification report=====")
print(classification_report(y_test,y_pred_rf))

import joblib


joblib.dump(rf_model, "heart_model.pkl", compress=3)
print("Random forest model saved succesfully!")






#SUPPORT VECTOR MECHINE

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

svm_model = SVC(kernel="linear", random_state=42)

svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\n=====support vector mechine(svm)=====")
print("accuracy", svm_accuracy)

print("\n=====confusion matrix=====")
print(confusion_matrix(y_test, y_pred_svm))

print("\n=====classificaton report=====")
print(classification_report(y_test, y_pred_svm))


#ACCURACY COMPARISION

# ==========================================
# ACCURACY COMPARISON
# ==========================================

results = {
    "Algorithm": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "SVM"
    ],
    "Accuracy": [
        accuracy,
        knn_accuracy,
        dt_accuracy,
        rf_accuracy,
        svm_accuracy
    ]
}

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df)

#BEST MODEL

# BEST MODEL

best_model = results_df.loc[results_df["Accuracy"].idxmax()]

print("\n===== BEST MODEL =====")

print("Algorithm:", best_model["Algorithm"])
print("Accuracy :", best_model["Accuracy"])


#CONCLUSION

# Randam forest acheived highest accuracy
# among all mechine learning alogarithms
# Therefore, random forest selected as
# the final model heart disease prediction













