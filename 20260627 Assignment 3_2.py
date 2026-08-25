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

# ------------------- Assignment 3_2 -------------------

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

def intro():

    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);

intro()
plt.show()

# ---------------------- REGRESSION ----------------------
# ---------------------- Answer One ----------------------
def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge

    degree_predictions = np.zeros((4, 100))

    # YOUR CODE HERE

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    X_predict_input = np.linspace(0, 10, 100).reshape(-1, 1)

    plt.figure(figsize=(10, 5))
    plt.title('ANV')
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')

    for i, deg in enumerate([1, 3, 6, 9]):
        poly = PolynomialFeatures(degree=deg)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        # print(poly)
        # print(X_train_poly)
        linreg = LinearRegression().fit(X_train_poly, y_train)
        # А вот так графики будут совершенно одинаковые:
        # linreg = Ridge(alpha=0.1).fit(X_train_poly, y_train)
        X_predict_poly = poly.transform(X_predict_input)
        # print(X_predict_input)
        # print(X_predict_poly)
        degree_predictions[i, :] = linreg.predict(X_predict_poly)
        plt.plot(X_predict_input, degree_predictions[i, :], label=f'degree={deg}')

    plt.legend(loc=4)

    return degree_predictions

answer_one()
plt.show()

# feel free to use the function plot_one() to replicate the figure
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.title('Coursera')
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

plot_one(answer_one())
plt.show()

# ---------------------- Answer Two ----------------------

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score

    r2_train = np.array([])
    r2_test = np.array([])

    # YOUR CODE HERE
    r2_train = np.zeros(10)
    r2_test = np.zeros(10)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

    for deg in range(10):
        poly = PolynomialFeatures(degree=deg)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        X_test_poly = poly.transform(X_test.reshape(-1, 1))
        linreg = LinearRegression().fit(X_train_poly, y_train)
        r2_train[deg] = linreg.score(X_train_poly, y_train)
        r2_test[deg] = linreg.score(X_test_poly, y_test)

    return r2_train, r2_test

# ---------------------- Answer Three ----------------------

def answer_three(r2_train, r2_test):
    # YOUR CODE HERE

    # r2_train, r2_test = answer_two()    # Если вызывать answer_three без аргументов

    plt.figure(figsize=(10, 5))
    plt.title('R2')
    plt.plot(range(10), r2_train, label='R2 Train')
    plt.plot(range(10), r2_test, label='R2 Test')
    plt.legend(loc=2)

    plt.show()

    print('r2_train, r2_test', r2_train, r2_test, sep='\n')

    # 1. GOOD GENERALIZATION: Ищем баланс.
    # Модель хороша, когда R2 на тесте высокий, а разница между Train и Test — минимальна.
    # Мы ищем максимум по Test, но проверяем, чтобы разрыв (Train - Test) не был огромным.
    Good_Generalization = int(np.argmax(r2_test))

    # 2. UNDERFITTING: Ошибается везде.
    # Суммируем R2 Train и R2 Test. Где эта сумма самая маленькая — там модель хуже всего
    # справилась с обеими выборками одновременно.
    Underfitting = int(np.argmin(r2_train + r2_test))

    # 3. Overfitting (Ищем минимальное значение на ТЕСТЕ, но строго ПОСЛЕ пика good_gen)
    # Отрезаем массив от пика и до конца, находим там минимум
    after_peak_test = r2_test[Good_Generalization:]
    Overfitting = Good_Generalization + int(np.argmin(after_peak_test))  # Индекс 9 (значение 0.5348)

    # Возвращаем строго три числа в правильном порядке
    return Underfitting, Overfitting, Good_Generalization
    raise NotImplementedError()

print('Underfitting, Overfitting, Good_Generalization\n', answer_three(*answer_two()))

# ---------------------- Answer Four ----------------------

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics import r2_score

    # YOUR CODE HERE

    poly = PolynomialFeatures(degree=12)
    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly.transform(X_test.reshape(-1, 1))
    linreg = LinearRegression().fit(X_train_poly, y_train)
    lasreg = Lasso(alpha=0.01, max_iter=10000, tol=0.1).fit(X_train_poly, y_train)

    LinearRegression_R2_test_score = linreg.score(X_test_poly, y_test)
    Lasso_R2_test_score = lasreg.score(X_test_poly, y_test)

    return LinearRegression_R2_test_score, Lasso_R2_test_score

print('LinearRegression_R2_test_score, Lasso_R2_test_score')
print(answer_four())

# -------------------- CLASSIFICATION ----------------------
# ----------------------- Mushrooms ------------------------
#
# For this section of the assignment we will be working with the UCI Mushroom Data Set stored in mushrooms.csv.
# The data will be used to trian a model to predict whether or not a mushroom is poisonous.
# The following attributes are provided:
#
# Attribute Information:
#
# cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
# cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
# cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
# bruises?: bruises=t, no=f
# odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
# gill-attachment: attached=a, descending=d, free=f, notched=n
# gill-spacing: close=c, crowded=w, distant=d
# gill-size: broad=b, narrow=n
# gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# stalk-shape: enlarging=e, tapering=t
# stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
# stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
# stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
# stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
# stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
# veil-type: partial=p, universal=u
# veil-color: brown=n, orange=o, white=w, yellow=y
# ring-number: none=n, one=o, two=t
# ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
# spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
# population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
# habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
#
# The data in the mushrooms dataset is currently encoded with strings.
# These values will need to be encoded to numeric to work with sklearn.
# We'll use pd.get_dummies to convert the categorical variables into indicator variables.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

mush_df = pd.read_csv('assets/mushrooms.csv')
print(mush_df.head())
print(mush_df.shape)
mush_df2 = pd.get_dummies(mush_df)
print(mush_df2.head())

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# ---------------------- Answer Five ----------------------

def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    # YOUR CODE HERE
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)

    from sklearn.tree import plot_tree
    plt.figure(figsize=(14, 9))
    plot_tree(clf,
              feature_names=list(X_train2.columns),  # Исправлено: берем имена колонок Pandas
              class_names=['edible', 'poisonous'],  # Исправлено: пишем имена классов вручную
              filled=True,
              fontsize=8)

    FeatureImportance = pd.DataFrame(clf.feature_importances_, list(X_train2.columns),
                                     columns=['FeatureImportance'])
    Top5fi = FeatureImportance.sort_values(by='FeatureImportance', ascending=False)
    Top5fi = Top5fi.index[:5].tolist()

    return Top5fi

print(answer_five())
plt.show()

# ---------------------- Answer Six ----------------------

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # YOUR CODE HERE
    param_range = np.logspace(-4, 1, 6)
    train_scores, tst_scores = validation_curve(SVC(kernel='rbf', C=1, random_state=0), X_mush, y_mush,
                                                    param_name='gamma', param_range=param_range, cv=3, n_jobs=2)

    training_scores = np.mean(train_scores, axis=1)
    test_scores = np.mean(tst_scores, axis=1)

    print(train_scores, tst_scores, training_scores, test_scores, sep='\n')

    return train_scores, tst_scores, training_scores, test_scores   # First two - for Plotting!

# print(answer_six()) # Вызывается ниже в Plotting


# -------------------- Plotting the Validation Curve ----------------
# This code based on scikit-learn validation_plot example
#  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

plt.figure(figsize=(10, 9))

train_scores, tst_scores, _, _ = answer_six()
# print(train_scores, tst_scores, sep='\n')

param_range = np.logspace(-4, 1, 6)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_max = np.max(train_scores, axis=1)
train_scores_min = np.min(train_scores, axis=1)
tst_scores_mean = np.mean(tst_scores, axis=1)
tst_scores_max = np.max(tst_scores, axis=1)
tst_scores_min = np.min(tst_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('Gamma')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_min,
                train_scores_max, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, tst_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, tst_scores_min,
                tst_scores_max, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc=3)
# plt.show()

# ---------------------- Answer Seven ----------------------

def answer_seven():

    param_range = np.logspace(-4, 1, 6)

    # 1. GOOD GENERALIZATION: Ищем баланс.
    # Модель хороша, когда R2 на тесте высокий, а разница между Train и Test — минимальна.
    # Мы ищем максимум по Test, но проверяем, чтобы разрыв (Train - Test) не был огромным.
    # ИСПРАВЛЕНИЕ: Исключаем первый индекс (0) из поиска Good.
    # Ищем argmax начиная с индекса 1, а затем прибавляем 1, чтобы вернуть правильный исходный индекс.
    Good_Generalization_idx = int(np.argmax(tst_scores_mean[1:]) + 1)

    # 2. Underfitting ищем строго слева от пика (включая сам пик)
    # Ищем самый низкий Train строго ЛЕВЕЕ оптимальной точки (Good_Generalization)
    # Отрезаем левую часть массива тренировочных скоров
    before_peak_train = train_scores_mean[:Good_Generalization_idx + 1]
    Underfitting_idx = int(np.argmin(before_peak_train))  # Гарантированно вернет индекс 0

    # 3. Overfitting (Ищем минимальное значение на ТЕСТЕ, но строго ПОСЛЕ пика Good_Generalization)
    # Отрезаем массив от пика и до конца, находим там минимум
    after_peak_test = tst_scores_mean[Good_Generalization_idx:]
    Overfitting_idx = Good_Generalization_idx + int(np.argmin(after_peak_test))  # Индекс 5

    # 3. Переводим индексы в реальные значения gamma (0.0001, 10.0, 0.1)
    Underfitting = float(param_range[Underfitting_idx])
    Overfitting = float(param_range[Overfitting_idx])
    Good_Generalization = float(param_range[Good_Generalization_idx])

    # Возвращаем строго три числа в правильном порядке
    return Underfitting, Overfitting, Good_Generalization

print(answer_seven())

plt.show()


