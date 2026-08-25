import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind

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


# ---------------------- Answer One ----------------------

def answer_one():

    df = pd.read_csv('assets/fraud_data.csv')
    print(df.head())

    fraud_percentage = df[df['Class'] == 1]['Class'].count() / df.shape[0]

    # Alternative count:
    # print(df[df['Class'] == 1].shape[0])
    print(df[df['Class'] == 1]['Class'].count())
    print(df.shape[0])
    return fraud_percentage

print(answer_one())


# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('assets/fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ---------------------- Answer One ----------------------

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, recall_score

    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    # Therefore the dummy 'most_frequent' classifier always predicts class 0
    y_dummy_pred = dummy_majority.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_dummy_pred)
    test_recall = recall_score(y_test, y_dummy_pred)

    return test_accuracy, test_recall

print(answer_two())


# ---------------------- Answer One ----------------------

def answer_three():
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    from sklearn.svm import SVC

    svm = SVC().fit(X_train, y_train)
    y_svm_pred = svm.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_svm_pred)
    test_recall = recall_score(y_test, y_svm_pred)
    test_precision = precision_score(y_test, y_svm_pred)

    return test_accuracy, test_recall, test_precision

print(answer_three())


# ---------------------- Answer Four ----------------------

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    grid_values = {'C': 1e9, 'gamma': 1e-07}
    svm = SVC(**grid_values).fit(X_train, y_train)
    y_svm_scores = svm.decision_function(X_test)
    print(y_svm_scores)

    y_svm_pred1 = svm.predict(X_test)
    print(y_svm_pred1)

    confusion1 = confusion_matrix(y_test, y_svm_pred1)
    print(confusion1)

    # сдвигаем threshold на -220
    y_svm_pred2 = (y_svm_scores > -220).astype(int)
    print(y_svm_pred2)

    confusion2 = confusion_matrix(y_test, y_svm_pred2)
    print(confusion2)

print(answer_four())


# ---------------------- Answer Five ----------------------

def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 1. Обучение модели
    lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
    y_lr_scores = lr.decision_function(X_test)

    # 2. Метрики Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_lr_scores)

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.gca().set_aspect('equal')

    # Находим recall при precision == 0.75
    idx_pr = np.argmin(np.abs(precision - 0.75))
    recall_value = recall[idx_pr]

    plt.plot(0.75, recall_value, 'x', markersize=12, fillstyle='none', c='r', mew=3)

    # 3. Метрики ROC кривой
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_lr_scores)

    roc_auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.gca().set_aspect('equal')

    # Ищем минимальное отклонение от 0.16
    min_diff = np.min(np.abs(fpr_lr - 0.16))
    print(min_diff)
    # Находим ВСЕ индексы, где FPR максимально близко к 0.16
    closest_indices = np.where(np.abs(fpr_lr - 0.16) == min_diff)
    print(closest_indices)
    # Забираем НАИБОЛЬШИЙ TPR среди этих точек (как требует условие задачи)
    true_positive_rate = np.max(tpr_lr[closest_indices])

    plt.plot(0.16, true_positive_rate, 'o', markersize=1, fillstyle='none', c='r', mew=3)
    plt.show()

    return recall_value, true_positive_rate

print(answer_five())


# ---------------------- Answer Six ----------------------

def answer_six():

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    import warnings
    # Игнорируем предупреждения о будущих изменениях в библиотеках
    warnings.filterwarnings('ignore')

    grid_values = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10]
    }

    lr = LogisticRegression(solver='liblinear')
    grid_lr_recall = GridSearchCV(lr, param_grid=grid_values, scoring='recall', cv=3)
    grid_lr_recall.fit(X_train, y_train)

    # Достаем плоский массив оценок напрямую
    scores = grid_lr_recall.cv_results_['mean_test_score']
    # Прямой перевод в 4 строки и 2 столбца (так требует внутренний робот Coursera)
    final = scores.reshape(4, 2)

    # 2. Выводим параметры и их оценки построчно для сравнения
    params = grid_lr_recall.cv_results_['params']
    scores = grid_lr_recall.cv_results_['mean_test_score']

    print("Порядок параметров в плоском массиве:")
    print("-" * 50)
    for p, score in zip(params, scores):
        print(f"Параметры: C={p['C']:<4} penalty={p['penalty']:<4} | Средний Recall: {score:.8f}")

    print(scores)

    return final

print(answer_six())







