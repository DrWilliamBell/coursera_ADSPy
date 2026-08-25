import pandas as pd
import numpy as np

# Снимаем ограничение на количество столбцов
pd.set_option('display.max_columns', 25)
# Увеличиваем общую ширину вывода (чтобы столбцы не переносились на новую строку)
pd.set_option('display.width', 200)
# Снимает ограничение на ширину каждого столбца
pd.set_option("display.max_colwidth", None)

# Возврат:
# pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows')
# pd.reset_option("display.max_colwidth")

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.DESCR) # Print the data set description
print(cancer.keys())
print(cancer)

# Q0
print(len(cancer['data'][0]))

#Q1
df = pd.DataFrame(cancer['data'])
df[30] = cancer['target']
df.columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension',
    'target']
df.index = pd.RangeIndex(start=0, stop=569, step=1)
print(df)

#Q2
# 1st method:
# target = df['target'].value_counts()
#2nd method:
target = df.groupby('target').size()
print(target)
# 0 'malignant'
# 1 'benign'
target.index = ['malignant', 'benign']
print(target)

#Q3
# For this example, we use 30 features for X
X = df.iloc[:, :-1]
y = df['target']
print(X.shape)
print(y.shape)

#Q4
from sklearn.model_selection import train_test_split
# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Q5
# Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier
# with X_train, y_train and using one nearest neighbor (n_neighbors = 1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
print(knn)
print(type(knn))

# Q6
# Using your knn classifier, predict the class label using the mean value for each feature
# 1 : lines and -1 : Auto-columns calc (here 30)
# ex1 = df.mean()[:-1].values.reshape(1, -1)
# print(ex1)
# Но так новые PyCharm и JN выдадут предупреждения, поэтому
ex1 = df.mean()[:-1]

# Но новые PyCharm и JN выдадут предупреждения, т.к. обучение было на DF, поэтому
# Превращаем Series в двумерный DataFrame (сохраняя имена колонок)
# Метод .to_frame().T делает из вертикального списка горизонтальную строку
ex1_df = ex1.to_frame().T
print(ex1_df)

cancer_prediction = knn.predict(ex1_df)
print(cancer_prediction)
print(type(cancer_prediction))

# Q7
# Using your knn classifier, predict the class labels for the test set X_test
X_test_prediction = knn.predict(X_test)
print(X_test_prediction)
print(X_test_prediction.shape)

# Q8
# Find the score (mean accuracy) of your knn classifier using X_test and y_test
print(knn.score(X_test, y_test))

# Q9 Optional Plot
# Try using the plotting function below to visualize the different predicition scores
# between train and test sets, as well as malignant and benign cells.

def accuracy_plot():
    import matplotlib.pyplot as plt

    Train_score = knn.score(X_train, y_train)
    Test_score = knn.score(X_test, y_test)

    # Use bolean masking y_train to filter X-train by True/False from y_train
    # BECAUSE X_train and y_train are syncronized in order after SPLIT
    train_mal_score = knn.score(X_train[y_train == 0], y_train[y_train == 0])
    train_ben_score = knn.score(X_train[y_train == 1], y_train[y_train == 1])

    # Same as for train - use boolean maskiing
    test_mal_score = knn.score(X_test[y_test == 0], y_test[y_test == 0])
    test_ben_score = knn.score(X_test[y_test == 1], y_test[y_test == 1])

    scores = [train_mal_score, train_ben_score, test_mal_score, test_ben_score]
    labels = ['Malignant\nTrain', 'Benign\nTrain', 'Malignant\nTest', 'Benign\nTest']

    plt.figure(figsize=(8, 5))
    plt.bar(labels, scores, color=['blue', 'blue', 'red', 'red'])

    # Оформление графика
    plt.title('Accuracy Scores by Dataset and Cell Type')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.19)  # Чтобы график не упирался в потолок

    # Добавим отображение точных цифр над столбцами для наглядности
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')

    plt.show()


accuracy_plot()




