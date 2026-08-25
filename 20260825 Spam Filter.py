import pandas as pd
import numpy as np

# Снимаем ограничение на количество столбцов
pd.set_option('display.max_columns', 25)
# Увеличиваем общую ширину вывода (чтобы столбцы не переносились на новую строку)
pd.set_option('display.width', 200)
# Снимает ограничение на ширину каждого столбца
pd.set_option("display.max_colwidth", 150)

# Возврат:
# pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows')
# pd.reset_option("display.max_colwidth")


# ============== 20260824: Assignment 4_3 ===============
#
# In this assignment you will explore text message data
# and create models to predict if a message is spam or not.
#
# --------------------- Data Prep -----------------------

# spam_data = pd.read_csv('assets/spam.csv')
#
# spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
# print(spam_data)
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],
#                                                     spam_data['target'],
#                                                     random_state=0)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ====================== ANV EMAILS ANALYSIS =======================

# 1. Читаем весь файл в одну строку
with open('assets/anv_emails_raw.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

# 2. Делим текст по слову "SEPARATOR"
# Функция .strip() уберет лишние пробелы и переносы строк по краям каждого письма
# плюс мы убираем \n, убираем \t (заменяем пробелами)
email_list = [email.replace('\n', ' ').replace('\t', ' ').strip() for email in raw_text.split('SEPARATOR')]

# 3. Убираем пустые элементы (если SEPARATOR стоял в самом начале или конце файла)
email_list = [email for email in email_list if email]

# 4. Создаем pandas Series
anv_emails = pd.Series(email_list, name="anv_emails")

print(anv_emails)
print(anv_emails[17][-150:])
print(anv_emails[20][-150:])

y_test_raw = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y_test = pd.Series(y_test_raw)
print(y_test)

spam_data = pd.DataFrame({'text': anv_emails,'target': y_test})
print(spam_data)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],
                                                    spam_data['target'],
                                                    random_state=0)


# X_test = anv_emails
# y_test_raw = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# y_test = pd.Series(y_test_raw)
# print(y_test)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Question 1
print('\nQ1')
# What percentage of the documents in spam_data are spam?
# This function should return a float, the percent value (i.e.  𝑟𝑎𝑡𝑖𝑜∗100).

def answer_one():

    print(spam_data.shape)
    spam_count = (spam_data['target'].sum() / len(spam_data)) * 100
    print('spam count %:')

    return spam_count

#print(answer_one())

# Question 2
print('\nQ2')
# Fit the training data X_train using a Count Vectorizer with default parameters.
# What is the longest token in the vocabulary?
# This function should return a string.

from sklearn.feature_extraction.text import CountVectorizer

def answer_two():

    vect = CountVectorizer().fit(X_train)
    print('len: ', len(vect.get_feature_names_out()))
    print('longest token:')
    longest_token = max(vect.get_feature_names_out(), key=len)

    return longest_token

#print(answer_two())

# Question 3
print('\nQ3')
# Fit and transform the training data X_train using a Count Vectorizer with default parameters.
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1.
# Find the area under the curve (AUC) score using the transformed test data.
# This function should return the AUC score as a float.

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():

    vect = CountVectorizer().fit(X_train)

    X_train_vectorized = vect.transform(X_train)
    print('len: ', len(vect.get_feature_names_out()))
    X_test_vectorized = vect.transform(X_test)
    # model CANNOT be trained on text data, we need to transform it to vector matrix,
    # where column heads are features and values are: how many times each feature occurs in the text
    print('X_train_vectorized:\n', X_train_vectorized[:1], '\n', X_train_vectorized[-1:])
    print('shape:', X_train_vectorized.shape)
    clfrNB = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    print(clfrNB)
    predictions = clfrNB.predict_proba(X_test_vectorized)
    auc = roc_auc_score(y_test, predictions[:, 1])
    # Смотрим что за слово с конкретным индексом:
    feature_names = vect.get_feature_names_out()
    print(f'Индекс 23: {feature_names[23]}')
    # Печатаем Feature Names для 1 и последней строк:
    # Находим индексы ненулевых колонок для ПЕРВОЙ строки
    first_row_indices = X_train_vectorized[0].indices
    # Находим индексы ненулевых колонок для ПОСЛЕДНЕЙ строки
    last_row_indices = X_train_vectorized[-1].indices
    print(f'First Line Feature Names:\n{feature_names[first_row_indices]}')
    print(f'Last Line Feature Names:\n{feature_names[last_row_indices]}\n')

    return auc

print(answer_three())

# Question 4
print('\nQ4')
# Fit and transform the training data X_train using a Tfidf Vectorizer with default parameters.
# The transformed data will be a compressed sparse row matrix where the number of rows
# is the number of documents in X_train, the number of columns is the number of features
# found by the vectorizer in each document, and each value in the sparse matrix is the tf-idf value.
# First find the max tf-idf value for every feature.
#
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf among the max tf-idf values?
#
# Put these features in two series where each series is sorted by tf-idf value.
# The index of the series should be the feature name, and the data should be the tf-idf.
#
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first,
# the list of 20 features with largest tf-idfs should be sorted largest first.
# Any entries with identical tf-ids should appear in lexigraphically increasing order
# by their feature name in boh series. For example,
# if the features "a", "b", "c" had the tf-idfs 1.0, 0.5, 1.0 in the series with the largest tf-idfs,
# then they should occur in the returned result in the order "a", "c", "b" with values 1.0, 1.0, 0.5.
#
# This function should return a tuple of two series (smallest tf-idfs series, largest tf-idfs series).

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():

    vect = TfidfVectorizer().fit(X_train)
    print('\nFeatures number (vect len): ', len(vect.get_feature_names_out()))
    X_train_vectorized = vect.transform(X_train)
    print('\nX_train_vectorized:\n', X_train_vectorized[:1])

    tfidf_max = X_train_vectorized.max(0).toarray()[0]
    print('\ntfidf_max:\n', tfidf_max)
    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    print('\nsorted_tfidf_index:\n', sorted_tfidf_index)

    feature_names = vect.get_feature_names_out()
    Smallest_tfidf = feature_names[sorted_tfidf_index[:20]]
    Largest_tfidf = feature_names[sorted_tfidf_index[:-21:-1]]
    print('\nSmallest 20 tfidf: \n', Smallest_tfidf, '\n')
    print('Largest 20 tfidf: \n', Largest_tfidf, '\n')

    print(f'CHECK: Индекс 7350: {feature_names[7350], tfidf_max[7350]}')

    print('\ntfidf_max 20 smallest:\n', tfidf_max[sorted_tfidf_index[:20]])
    print('\ntfidf_max 20 largest:\n', tfidf_max[sorted_tfidf_index[:-21:-1]])

    Smallest = pd.Series(tfidf_max[sorted_tfidf_index[:20]], index=Smallest_tfidf)
    Largest = pd.Series(tfidf_max[sorted_tfidf_index[:-21:-1]], index=Largest_tfidf)
    Smallest = Smallest.sort_index(ascending=True).sort_values(ascending=True)
    Largest = Largest.sort_index(ascending=True).sort_values(ascending=False)
    print('Smallest:\n', Smallest, '\n')
    print('Largest:\n', Largest, '\n')
    # ЭТОТ РЕЗУЛЬТАТ МОЖЕТ НЕ ПРОЙТИ В COURSERA!


    # ++++++++++++ Для ТУПОГО древнего coursera (цепочки не работают) +++++++++++++
    #
    # Однако это более правильное решение, т.к. например для 19 и 20 значений min_series
    # по алфавиту с такими же tfidf будут другие токены (начинающиеся на 'a') !!!
    df = pd.DataFrame({'tfidf': tfidf_max, 'feature': feature_names})
    print(df)

    # === 1. СЕРИЯ С НАИМЕНЬШИМИ TF-IDF (Топ-20 минимальных) ===
    # Сортируем: tfidf по возрастанию (True), feature по алфавиту (True)
    min_sorted = df.sort_values(by=['tfidf', 'feature'], ascending=[True, True])
    # Берем первые 20 и превращаем обратно в Series
    # (.values нужно чтобы избавиться от старых индексов df и потенциальных проблем с этим)
    min_series = pd.Series(min_sorted['tfidf'].values, index=min_sorted['feature']).head(20)
    # Можно ещё вот так:
    # Делаем колонку 'feature' индексом и сразу вытаскиваем 'tfidf' как чистый Series!
    # min_series = min_sorted.set_index('feature')['tfidf'].head(20)

    # === 2. СЕРИЯ С НАИБОЛЬШИМИ TF-IDF (Топ-20 максимальных) ===
    # Сортируем: tfidf по УБЫВАНИЮ (False), а feature ВСЁ РАВНО по алфавиту (True)
    max_sorted = df.sort_values(by=['tfidf', 'feature'], ascending=[False, True])
    # Берем первые 20 и превращаем обратно в Series
    max_series = pd.Series(max_sorted['tfidf'].values, index=max_sorted['feature']).head(20)
    # Можно ещё вот так:
    # Делаем колонку 'feature' индексом и сразу вытаскиваем 'tfidf' как чистый Series!
    # max_series = max_sorted.set_index('feature')['tfidf'].head(20)

    return min_series, max_series

#print(answer_four())

# Question 5
print('\nQ5')
# Fit and transform the training data X_train using a Tfidf Vectorizer
# ignoring terms that have a document frequency strictly lower than 3.
#
# Then fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1
# and compute the area under the curve (AUC) score using the transformed test data.
#
# This function should return the AUC score as a float.

def answer_five():

    vect = TfidfVectorizer(min_df=3).fit(X_train)
    print('\nFeatures number (vect len): ', len(vect.get_feature_names_out()))
    X_train_vectorized = vect.transform(X_train)
    print('\nX_train_vectorized:\n', X_train_vectorized[:1])
    print('shape:', X_train_vectorized.shape)
    X_test_vectorized = vect.transform(X_test)
    # model CANNOT be trained on text data, we need to transform it to vector matrix,
    # where column heads are features and values are: how many times each feature occurs in the text

    clfrNB = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    print(clfrNB)
    predictions = clfrNB.predict_proba(X_test_vectorized)
    print('predictions:\n', predictions, '\n')
    auc = roc_auc_score(y_test, predictions[:, 1])

    return auc

print(answer_five())

# Question 6
print('\nQ6')
# What is the average length of documents (number of characters) for not spam and spam documents?
#
# This function should return a tuple (average length not spam, average length spam).

def answer_six():

    avg_len_not_spam = spam_data[spam_data['target'] == 0]['text'].str.len().mean()
    avg_len_spam = spam_data[spam_data['target'] == 1]['text'].str.len().mean()

    return avg_len_not_spam, avg_len_spam

#print(answer_six())

# =================================================================================================
# The following function has been provided to help you combine new features into the training data:
#

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# Question 7
print('\nQ7')
# Fit and transform the training data X_train using a Tfidf Vectorizer
# ignoring terms that have a document frequency strictly lower than 5.
#
# Using this document-term matrix and an additional feature, the length of document (number of characters),
# fit a Support Vector Classification model with regularization C=10000.
# Then compute the area under the curve (AUC) score using the transformed test data.
#
# Hint: Since probability is set to false, use the model's decision_function
# on the test data when calculating the target scores to use in roc_auc_score
#
# This function should return the AUC score as a float.

from sklearn.svm import SVC

def answer_seven():

    vect = TfidfVectorizer(min_df=5).fit(X_train)
    print('\nFeatures number (vect len): ', len(vect.get_feature_names_out()))
    X_train_vectorized = vect.transform(X_train)
    print('\nX_train_vectorized:\n', X_train_vectorized[:1])
    print('shape:', X_train_vectorized.shape)
    X_test_vectorized = vect.transform(X_test)

    # 1. Считаем длины (просто списки чисел)
    train_len = [len(text) for text in X_train]
    test_len = [len(text) for text in X_test]
    print('\nX-train type: ',type(X_train))

    # 2. Передаем в функцию старую матрицу и наш список чисел
    X_train_enhanced = add_feature(X_train_vectorized, train_len)
    X_test_enhanced = add_feature(X_test_vectorized, test_len)
    print('\nX_train_enhanced (vectorized and add_feature):\n', X_train_enhanced[:1])
    print('shape:', X_train_enhanced.shape)

    clfrSVM = SVC(C=10000).fit(X_train_enhanced, y_train)
    print(clfrSVM)
    y_scores = clfrSVM.decision_function(X_test_enhanced)
    print('y_scores:\n', y_scores, '\n')
    auc = roc_auc_score(y_test, y_scores)

    return auc

print(answer_seven())

# Question 8
print('\nQ8')
# What is the average number of digits per document for not spam and spam documents?
#
# Hint: Use \d for digit class
#
# This function should return a tuple (average # digits not spam, average # digits spam).

def answer_eight():

    avg_dig_not_spam = spam_data[spam_data['target'] == 0]['text'].str.count(r'\d').mean()
    avg_dig_spam = spam_data[spam_data['target'] == 1]['text'].str.count(r'\d').mean()

    # Интересное решение 2:
    print(spam_data.groupby('target')['text'].apply(lambda x: x.str.count(r'\d').mean()))

    return avg_dig_not_spam, avg_dig_spam

#print(answer_eight())

# Question 9
print('\nQ9')
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms
# that have a document frequency strictly lower than 5 and using word n-grams from n=1 to n=3
# (unigrams, bigrams, and trigrams).
#
# Using this document-term matrix and the following additional features:
#
# the length of document (number of characters)
# number of digits per document
# fit a Logistic Regression model with regularization C=100 and max_iter=1000.
# Then compute the area under the curve (AUC) score using the transformed test data.
#
# This function should return the AUC score as a float.

from sklearn.linear_model import LogisticRegression

def answer_nine():

    vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    print('\nFeatures number (vect len): ', len(vect.get_feature_names_out()))
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    print('\nX_train_vectorized:\n', X_train_vectorized[:1])
    print('shape:', X_train_vectorized.shape)

    # 1. Считаем длины (просто списки чисел)
    train_len = [len(text) for text in X_train]
    test_len = [len(text) for text in X_test]
    # 2. Считаем длины (просто списки чисел)
    # метод .count для списков не умеет работать с re
    train_dig = X_train.str.count(r"\d")
    test_dig = X_test.str.count(r"\d")
    print('train_dig:\n', train_dig)

    # 3. Передаем в функцию старую матрицу и наши списки чисел
    X_train_enhanced = add_feature(X_train_vectorized, [train_len, train_dig])
    X_test_enhanced = add_feature(X_test_vectorized, [test_len, test_dig])
    print('\nX_train_enhanced (vectorized and add_feature):\n', X_train_enhanced[-4])
    print('shape:', X_train_enhanced.shape)

    clfrLR = LogisticRegression(C=100, max_iter=1000).fit(X_train_enhanced, y_train)
    print('\nclfrLR:\n', clfrLR)
    predictions = clfrLR.predict_proba(X_test_enhanced)
    print('\npredictions:\n', predictions, '\n')
    auc = roc_auc_score(y_test, predictions[:, 1])

    return auc

print(answer_nine())

# Question 10
print('\nQ10')
# What is the average number of non-word characters
# (anything other than a letter, digit or underscore) per document for not spam and spam documents?
#
# Hint: Use \w and \W character classes
#
# This function should return a tuple
# (average # non-word characters not spam, average # non-word characters spam).

def answer_ten():

    avg_nonword_not_spam = spam_data[spam_data['target'] == 0]['text'].str.count(r'\W').mean()
    avg_nonword_spam = spam_data[spam_data['target'] == 1]['text'].str.count(r'\W').mean()

    # Интересное решение 2:
    print(spam_data.groupby('target')['text'].apply(lambda x: x.str.count(r'\W').mean()))

    return avg_nonword_not_spam, avg_nonword_spam

#print(answer_ten())

# Question 11
print('\nQ11')
# Fit and transform the first 2000 rows of training data X_train using a Count Vectorizer
# ignoring terms that have a document frequency strictly lower than 5 and using character n-grams from n=2 to n=5.
#
# To tell Count Vectorizer to use character n-grams pass in analyzer='char_wb'
# which creates character n-grams only from text inside word boundaries.
# This should make the model more robust to spelling mistakes.
#
# Using this document-term matrix and the following additional features:
#
# the length of document (number of characters)
# number of digits per document
# number of non-word characters (anything other than a letter, digit or underscore.)
# fit a Logistic Regression model with regularization C=100 and max_iter=1000.
# Then compute the area under the curve (AUC) score using the transformed test data.
#
# Also find the 10 smallest and 10 largest coefficients from the model
# and return them along with the AUC score in a tuple.
#
# The list of 10 smallest coefficients should be sorted smallest first,
# the list of 10 largest coefficients should be sorted largest first.
#
# The three features that were added to the document term matrix should have the following names
# should they appear in the list of coefficients: ['length_of_doc', 'digit_count', 'non_word_char_count']
#
# This function should return a tuple (AUC score as a float, smallest coefs list, largest coefs list).

def answer_eleven():

    X_train_2k = X_train[:2000]
    y_train_2k = y_train[:2000]
    print(X_train_2k.shape)

    vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train_2k)
    print('vect: ', vect)
    print('\nFeatures number (vect len): ', len(vect.get_feature_names_out()))
    X_train_vectorized = vect.transform(X_train_2k)
    X_test_vectorized = vect.transform(X_test)
    print('\nX_train_vectorized shape:', X_train_vectorized.shape)

    # 1. Считаем длины (просто списки чисел)
    train_len = [len(text) for text in X_train_2k]
    test_len = [len(text) for text in X_test]
    # 2. Считаем цифры (просто списки чисел)
    # метод .count для списков не умеет работать с re
    train_dig = X_train_2k.str.count(r"\d")
    test_dig = X_test.str.count(r"\d")
    print('train_dig:\n', train_dig)
    # 3. Считаем non-word characters (просто списки чисел)
    train_nonword = X_train_2k.str.count(r"\W")
    test_nonword = X_test.str.count(r"\W")
    print('train_nonword:\n', train_nonword)

    # 4. Передаем в функцию старую матрицу и наши списки чисел
    X_train_enhanced = add_feature(X_train_vectorized, [train_len, train_dig, train_nonword])
    X_test_enhanced = add_feature(X_test_vectorized, [test_len, test_dig, test_nonword])
    print('\nX_train_enhanced (vectorized and add_feature):\n', X_train_enhanced[-4])
    print('shape:', X_train_enhanced.shape)

    clfrLR = LogisticRegression(C=100, max_iter=1000).fit(X_train_enhanced, y_train_2k)
    print('\nclfrLR:\n', clfrLR)
    predictions = clfrLR.predict_proba(X_test_enhanced)
    print('\npredictions:\n', predictions, '\n')
    auc = roc_auc_score(y_test, predictions[:, 1])
    print('AUC: ', auc)

    feature_names = np.array(vect.get_feature_names_out())
    # Добавляем названия ваших кастомных признаков В ТОМ ЖЕ ПОРЯДКЕ, как добавляли их в матрицу
    # Используем .extend() для добавления списка элементов (если list и по одному - то .append())
    feature_names = np.append(feature_names, ['length_of_doc', 'digit_count', 'non_word_char_count'])
    print('\nfeature names:\n', feature_names)

    print('clfrLR_coef_:\n', clfrLR.coef_)
    print('clfrLR_coef_[0]:\n', clfrLR.coef_[0])
    sorted_coef_index = clfrLR.coef_[0].argsort()
    print('sorted_coef_index:\n', sorted_coef_index)

    SmallestCoefs = feature_names[sorted_coef_index[:10]].tolist()
    LargestCoefs = feature_names[sorted_coef_index[:-11:-1]].tolist()
    print('Smallest Coefs:\n', SmallestCoefs)
    print('Largest Coefs:\n', LargestCoefs)

    return auc, SmallestCoefs, LargestCoefs

#print(answer_eleven())

#
# ---------------------------------------------------------------------------------------------------------------
#

# ====================== SPAM CHECKER FOR ANV REAL EMAILS =======================

# # 1. Читаем весь файл в одну строку
# with open('assets/anv_emails_raw.txt', 'r', encoding='utf-8') as file:
#     raw_text = file.read()
#
# # 2. Делим текст по слову "SEPARATOR"
# # Функция .strip() уберет лишние пробелы и переносы строк по краям каждого письма
# # плюс мы убираем \n, убираем \t (заменяем пробелами)
# email_list = [email.replace('\n', ' ').replace('\t', ' ').strip() for email in raw_text.split('SEPARATOR')]
#
# # 3. Убираем пустые элементы (если SEPARATOR стоял в самом начале или конце файла)
# email_list = [email for email in email_list if email]
#
# # 4. Создаем pandas Series
# anv_emails = pd.Series(email_list, name="anv_emails")
#
# print(anv_emails)
# print(anv_emails[17][-150:])
# print(anv_emails[20][-150:])
#
# X_test = anv_emails
# y_test_raw = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# y_test = pd.Series(y_test_raw)
# print(y_test)

X_train_2k = X_train.copy()
y_train_2k = y_train.copy()
print(X_train_2k.shape)

vect = CountVectorizer(min_df=5, ngram_range=(2, 5), analyzer='char_wb').fit(X_train_2k)
print('vect: ', vect)
print('\nFeatures number (vect len): ', len(vect.get_feature_names_out()))
X_train_vectorized = vect.transform(X_train_2k)
X_test_vectorized = vect.transform(X_test)
print('\nX_train_vectorized shape:', X_train_vectorized.shape)

# 1. Считаем длины (просто списки чисел)
train_len = [len(text) for text in X_train_2k]
test_len = [len(text) for text in X_test]
# 2. Считаем цифры (просто списки чисел)
# метод .count для списков не умеет работать с re
train_dig = X_train_2k.str.count(r"\d")
test_dig = X_test.str.count(r"\d")
print('train_dig:\n', train_dig)
# 3. Считаем non-word characters (просто списки чисел)
train_nonword = X_train_2k.str.count(r"\W")
test_nonword = X_test.str.count(r"\W")
print('train_nonword:\n', train_nonword)

# 4. Передаем в функцию старую матрицу и наши списки чисел
X_train_enhanced = add_feature(X_train_vectorized, [train_len, train_dig, train_nonword])
X_test_enhanced = add_feature(X_test_vectorized, [test_len, test_dig, test_nonword])
print('\nX_train_enhanced (vectorized and add_feature):\n', X_train_enhanced[-4])
print('shape:', X_train_enhanced.shape)

clfrLR = LogisticRegression(C=100, max_iter=1000).fit(X_train_enhanced, y_train_2k)
print('\nclfrLR:\n', clfrLR)

# ----------------------------------------------------------------------------------------------------------------
# Раз вы уже используете predict_proba, значит, дело именно в неправильно выбранном пороге отсечения (threshold).
# Сейчас ваш код превращает вероятности в predictions по стандартному порогу 0.5, из-за чего все пограничные
# значения падают в 0. Но так как AUC = 0.91, модель отлично разделяет классы — вам просто нужно помочь ей
# провести черту в правильном месте.Вот как автоматически найти и применить идеальный порог прямо в вашем коде.
# A. Находим оптимальный порог через ROC-кривуюИспользуйте геометрический метод (индекс Юдена),
# чтобы найти порог, который дает лучший баланс между чувствительностью (Recall) и специфичностью:

from sklearn.metrics import roc_curve

# 1. Получаем вероятности спама (класс 1)
# Убедитесь, что берете именно второй столбец [:, 1]
probabilities = clfrLR.predict_proba(X_test_enhanced)[:, 1]

# 2. Считаем точки ROC-кривой
fpr, tpr, thresholds = roc_curve(y_test, probabilities)

# 3. Находим порог, максимизирующий (True Positive Rate - False Positive Rate)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Оптимальный порог для классификации: {optimal_threshold:.4f}")
print('НО ЭТО ВИДИМО ПОДГОН!')

# B. Применяем новый порог к предсказаниямЗамените стандартный вызов .predict()
# на ручное присвоение классов на основе найденного optimal_threshold:
# Вместо predictions = clfrLR.predict(X_test) делаем:
predictions = (probabilities >= optimal_threshold).astype(int)
# ----------------------------------------------------------------------------------------------------------------

# БЫЛО:
#predictions = clfrLR.predict_proba(X_test_enhanced)
print('\npredictions:\n', predictions, '\n')

auc = roc_auc_score(y_test, predictions)
print('AUC: ', auc)

feature_names = np.array(vect.get_feature_names_out())
# Добавляем названия ваших кастомных признаков В ТОМ ЖЕ ПОРЯДКЕ, как добавляли их в матрицу
# Используем .extend() для добавления списка элементов (если list и по одному - то .append())
feature_names = np.append(feature_names, ['length_of_doc', 'digit_count', 'non_word_char_count'])
print('\nfeature names:\n', feature_names)

print('clfrLR_coef_:\n', clfrLR.coef_)
print('clfrLR_coef_[0]:\n', clfrLR.coef_[0])
sorted_coef_index = clfrLR.coef_[0].argsort()
print('sorted_coef_index:\n', sorted_coef_index)

SmallestCoefs = feature_names[sorted_coef_index[:10]].tolist()
LargestCoefs = feature_names[sorted_coef_index[:-11:-1]].tolist()
print('Smallest Coefs:\n', SmallestCoefs)
print('Largest Coefs:\n', LargestCoefs)

# БЫЛО (нужно ли это?):
# Получаем бинарные предсказания от вашей модели
# (Убедитесь, что ваш X_test сначала проходит через векторизатор, если модель ожидает числа)
#predictions = clfrLR.predict(X_test_enhanced)

# Собираем всё в один DataFrame
df_results = pd.DataFrame({
    'text': X_test.values,         # Текст писем
    'y_test': y_test.values,       # Истинные метки (0 или 1)
    'predictions': predictions     # Предсказания модели (0 или 1)
})

# Выводим результат на экран
print('\nFINAL SPAM ANALYZER RESULTS:')
print(df_results)













