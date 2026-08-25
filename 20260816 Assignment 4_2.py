import pandas as pd
import sys

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


# =============== Assignment 2 - Introduction to NLTK ================
#
# In part 1 of this assignment you will use nltk to explore the CMU Movie Summary Corpus.
# All data is released under a Creative Commons Attribution-ShareAlike License.
# Then in part 2 you will create a spelling recommender function that uses nltk
# to find words similar to the misspelling.
#
# -------------- Part 1 - Analyzing Plots Summary Text ---------------

import nltk
import numpy as np

nltk.data.path.append("assets/")

# If you would like to work with the raw text you can use 'plots_raw':
with open('assets/plots.txt', 'rt', encoding="utf8") as f:
    plots_raw = f.read()

# If you would like to work with the plot summaries in nltk.Text format you can use 'text1':
plots_tokens = nltk.word_tokenize(plots_raw)
text1 = nltk.Text(plots_tokens)

print(plots_tokens[:100])
print(type(text1))
print(text1[:100])
print(" ".join(text1[:100]))
print()

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w, 'v') for w in text1]
print(lemmatized[:100])
print()


# Example 1
# How many tokens (words and punctuation symbols) are in text1?
#
# This function should return an integer.

def example_one():
    return len(nltk.word_tokenize(plots_raw))  # or alternatively len(text1)

print(example_one())


# Example 2
# How many unique tokens (unique words and punctuation) does text1 have?
#
# This function should return an integer.

def example_two():
    return len(set(nltk.word_tokenize(plots_raw)))  # or alternatively len(set(text1))

print(example_two())


# Example 3
# After lemmatizing the verbs, how many unique tokens does text1 have?
#
# This function should return an integer.

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

print(example_three())


# Question 1
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
#
# This function should return a float.

def answer_one():

    diversity = len(set(text1)) / len(text1)

    return diversity

print(answer_one())


# Question 2
# What percentage of tokens is 'love' or 'Love'?
#
# This function should return a float.

def answer_two():

    # --- Способ 1 ---
    from nltk import FreqDist
    # 1. Создаем счетчик по вашим токенам (делается 1 раз)
    fdist = FreqDist(plots_tokens)
    # 2. Мгновенно узнаем количество конкретных слов (регистр важен!)
    print('love', fdist['love'])  # Сколько раз встретилось слово 'love'
    print('Love', fdist['Love'])  # Сколько раз встретилось слово 'Love'

    # --- Способ 2 ---
    from collections import Counter
    # Создаем словарь частот
    word_counts = Counter(plots_tokens)
    # Быстро ищем нужные слова
    print('love', word_counts['love'])
    print('Love', word_counts['Love'])

    print('total tokens:', len(text1))
    print(((449 + 15) / 374441) * 100)

    # --- Способ 3 --- Standard Python List Method:
    count_love = plots_tokens.count('love') + plots_tokens.count('Love')

    return (count_love / len(plots_tokens)) * 100

print(answer_two())


# Question 3
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
#
# This function should return a list of 20 tuples where each tuple is of the form (token, frequency).
# The list should be sorted in descending order of frequency.

def answer_three():

    from nltk import FreqDist

    dist = FreqDist(text1)
    print(type(dist))
    vocab = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    vocab = vocab[:20]
    print(vocab)

    # built-in FreqDist method:
    top_20 = dist.most_common(20)
    # print(top_20)

    return top_20

print(answer_three())


# Question 4
# What tokens have a length of greater than 5 and frequency of more than 200?
#
# This function should return an alphabetically sorted list of the tokens that match the above constraints.
# To sort your list, use sorted()

def answer_four():

    from nltk import FreqDist

    dist = FreqDist(text1)
    vocab1 = dist.keys()
    freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 200]
    fw_sorted = sorted(freqwords)

    return fw_sorted

print(answer_four())


# Question 5
# Find the longest token in text1 and that token's length.
#
# This function should return a tuple (longest_word, length).

def answer_five():

    # долгий способ через цикл:
    w = ''
    for word in text1:
        if len(word) > len(w):
            w = word

    print(w, len(w))

    # max() сама найдет элемент с максимальной длиной:
    w = max(set(text1), key=len)

    return (w, len(w))

print(answer_five())


# Question 6
# What unique words have a frequency of more than 2000? What is their frequency?
#
# "Hint: you may want to use isalpha() to check if the token is a word and not punctuation."
#
# This function should return a list of tuples of the form (frequency, word) sorted in descending order of frequency.

def answer_six():

    from nltk import FreqDist

    dist = FreqDist(text1)
    vocab1 = dist.items()

    freqwords = [(x, w) for w, x in vocab1 if w.isalpha() and x > 2000]
    fw_sorted = sorted(freqwords, reverse=True)

    return fw_sorted

print(answer_six())


# Question 7
# text1 is in nltk.Text format that has been constructed using tokens output by nltk.word_tokenize(plots_raw).
#
# Now, use nltk.sent_tokenize on the tokens in text1 by joining them using whitespace to output
# a sentence-tokenized copy of text1. Report the average number of whitespace separated tokens per sentence
# in the sentence-tokenized copy of text1.
#
# This function should return a float.

def answer_seven():

    sentences = nltk.sent_tokenize(' '.join(text1))
    print(sentences[:10])

    tokens_in_sent = [len(s.split()) for s in sentences]
    ave_num = sum(tokens_in_sent) / len(sentences)

    return ave_num

print(answer_seven())


# Question 8
# What are the 5 most frequent parts of speech in text1? What is their frequency?
#
# This function should return a list of tuples of the form (part_of_speech, frequency)
# sorted in descending order of frequency.

def answer_eight():

    from nltk import FreqDist

    pos = nltk.pos_tag(text1)
    print(pos[:100])
    pos1 = [item[1] for item in pos]

    dist = FreqDist(pos1)
    pos_sorted = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    print(pos_sorted)

    return pos_sorted[:5]

print(answer_eight())


# ------------------ Part 2 - Spelling Recommender ------------------
#
# For this part of the assignment you will create three different spelling recommenders,
# that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
#
# For every misspelled word, the recommender should find find the word in correct_spellings
# that has the shortest distance*, and starts with the same letter as the misspelled word,
# and return that word as a recommendation.
#
# *Each of the three different recommenders will use a different distance measure (outlined below).
#
# Each of the recommenders should provide recommendations for the three default words provided:
# ['cormulent', 'incendenece', 'validrate'].


# import nltk               # Need just once to download 'words'!
# nltk.download('words')    # Need just once to download 'words'!

from nltk.corpus import words

correct_spellings = words.words()
print(correct_spellings[:10:])
print(len(correct_spellings))

# Question 9
# For this recommender, your function should provide recommendations for the three default words
# provided above using the following distance metric:
#
# Jaccard distance on the trigrams of the two words.
#
# Refer to:
#
# NLTK Jaccard distance
# NLTK ngrams
# This function should return a list of length three:
# ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    import pandas as pd
    from nltk.metrics import jaccard_distance
    from nltk.util import ngrams

    # ng_ent = [list(ngrams(w, 3)) for w in entries]
    # print(ng_ent)

    result = []

    for word in entries:

        # 1. Фильтруем словарь: берем только слова на ту же букву (в нижнем регистре)
        filtered_words = [w for w in correct_spellings if w.lower().startswith(word[0].lower())]

        # 2. Генерируем 3-граммы только для отфильтрованных слов
        ng_cs = [set(ngrams(w, 3)) for w in filtered_words]

        # 3. Создаем временный DataFrame для текущего слова
        df = pd.DataFrame({
            'word': filtered_words,
            'ngram': ng_cs})

        # 4. Генерируем 3-граммы для текущего слова и СРАЗУ переводим в set
        word_ng_set = set(ngrams(word, 3))

        # 5. Внутри apply считаем стандартный jaccard_distance
        df[word] = df['ngram'].apply(lambda x: jaccard_distance(x, word_ng_set))

        print(df)

        # 6. Находим индекс строки с минимальным расстоянием
        mindist_word_idx = df[word].idxmin()
        mindist_word = df.loc[mindist_word_idx, 'word']

        result.append(mindist_word)

    return result  # Вернет список исправленных слов, например ['corpulent', 'indecence', 'validate']

print(answer_nine())


# Question 9 ----------> ALTERNATIVE VERSION (Gemini)

def answer_nine_(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk.metrics.distance import jaccard_distance
    from nltk.util import ngrams

    # Предполагаем, что correct_spellings загружен из nltk.corpus.words.words()
    recommendations = []

    for entry in entries:
        # 1. Выбираем слова ТОЛЬКО на ту же букву, что и опечатка
        candidates = [w for w in correct_spellings if w[0].lower() == entry[0].lower()]

        # 2. Разделяем исходное слово на триграммы
        entry_ngrams = set(ngrams(entry, 3))

        # 3. Находим кандидата с МИНИМАЛЬНЫМ расстоянием Жаккара
        best_match = min(candidates, key=lambda w: jaccard_distance(entry_ngrams, set(ngrams(w, 3))))

        recommendations.append(best_match)

    return recommendations  # Вернет ['corpulent', 'indecence', 'validate']

print(answer_nine_())


# Question 10
# For this recommender, your function should provide recommendations for the three default
# words provided above using the following distance metric:
#
# Jaccard distance on the 4-grams of the two words.
#
# Refer to:
#
# NLTK Jaccard distance
# NLTK ngrams
# This function should return a list of length three:
# ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):

    from nltk.metrics.distance import jaccard_distance
    from nltk.util import ngrams

    # Предполагаем, что correct_spellings загружен из nltk.corpus.words.words()
    recommendations = []

    for entry in entries:
        # 1. Выбираем слова ТОЛЬКО на ту же букву, что и опечатка
        # кроме того, для ускорения вычислений отбрасываем слова менее Ngrams (4)
        candidates = [w for w in correct_spellings if w[0].lower() == entry[0].lower() and len(w) >= 4]

        # 2. Разделяем исходное слово на 4-граммы
        entry_ngrams = set(ngrams(entry, 4))

        # 3. Находим кандидата с МИНИМАЛЬНЫМ расстоянием Жаккара
        best_match = min(candidates, key=lambda w: jaccard_distance(entry_ngrams, set(ngrams(w, 4))))

        recommendations.append(best_match)

    return recommendations  # Вернет ['cormus', 'incendiary', 'valid']

print(answer_ten())


# Question 11
# For this recommender, your function should provide recommendations for the three
# default words provided above using the following distance metric:
#
# Edit distance on the two words with transpositions.
#
# Refer to:
#
# NLTK edit distance
# This function should return a list of length three:
# ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):

    from nltk.metrics.distance import edit_distance

    # Предполагаем, что correct_spellings загружен из nltk.corpus.words.words()
    recommendations = []

    for entry in entries:
        # 1. Выбираем слова ТОЛЬКО на ту же букву, что и опечатка
        candidates = [w for w in correct_spellings if w[0].lower() == entry[0].lower()]

        # 2. Находим кандидата с МИНИМАЛЬНЫМ расстоянием Жаккара
        best_match = min(candidates, key=lambda w: edit_distance(entry, w, transpositions=True))

        recommendations.append(best_match)

    return recommendations  # Вернет ['corpulent', 'intendence', 'validate']

print(answer_eleven())













