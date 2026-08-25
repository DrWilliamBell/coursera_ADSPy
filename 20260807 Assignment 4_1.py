import pandas as pd

# # Снимаем ограничение на количество столбцов
pd.set_option('display.max_columns', 25)
# # Увеличиваем общую ширину вывода (чтобы столбцы не переносились на новую строку)
pd.set_option('display.width', 250)
# # Снимает ограничение на ширину каждого столбца
pd.set_option("display.max_colwidth", 150)
# Выводить ВСЕ строки (без ограничений)
pd.set_option('display.max_rows', None)

# Возврат:
# pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows')
# pd.reset_option("display.max_colwidth")

# ---------------- IMPORTING CHAOTIC TXT ----------------

# Alternative way of text import to DF then to PD:
#
# text = pd.read_csv('assets/dates.txt', header=None, sep='\t', quoting=3, engine='python', names=['date'])
# (bad idea to use read_csv as autoparser is included and looking for separators, etc)
#
# colspecs=[(0, None)] принудительно отключает поиск колонок внутри строки:
# text = pd.read_fwf('assets/dates.txt', colspecs=[(0, None)], header=None, names=['date'])
# pd.read_csv_fwf cuts '\n'
# print(text)
# sr = pd.Series(text['date'])
# print(sr)

doc = []
with open('assets/dates.txt') as file:
    # print(file.read())
    # file.seek(0)
    for line in file:
        doc.append(line)        # line.strip('\n') - if we want DF to be identical to SR

df = pd.Series(doc)

df2 = df.copy()     # For Method 2
# print(df)


# If we want to compare Series1 and Series2 (sr vs df):
# print(df == sr)   # by lines
# as a whole:
# is_identical = df.equals(sr)
# print(is_identical)

# -------------------- CLEANING THE DATES ------------------
#
# ------------------------- Method 1 -----------------------

def date_sorter():
    import re
    order = None

    for idx, line in df.items():

        # 04/20/2009; 04/20/09; 4/20/09; 4/3/09; 4-13-89
        dt = re.findall(r'\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})', line)
        if dt:
            df[idx] = str(dt)
            continue

        # Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
        # 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
        # Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
        # Feb 2009; Sep 2009; Oct 2010
        dt = re.findall(r'(?:\d{1,2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,]*[ -]'
                       r'(?:\d{1,2}(?:st|nd|rd|th)?[, -]+)?\d{4}', line)
        if dt:
            df[idx] = str(dt)
            continue

        # 6/2008; 12/2009
        # 2009; 2010
        dt = re.findall(r'(?:\d{1,2}/)?\d{4}', line)
        if dt:
            df[idx] = str(dt)
            continue

    dt = pd.DataFrame()
    dt['extracted'] = df.str.extract(
        r'(\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})|(?:\d{1,2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,]*[ -]'
        r'(?:\d{1,2}(?:st|nd|rd|th)?[, -]+)?\d{4}|(?:\d{1,2}/)?(?<!\d{3}-)(?<!\d)\d{4}(?!\d))')

    # replace weekdays with 3 letter abbrevations:
    dt['corrected'] = dt['extracted'].str.replace(
        r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,]*)',
        lambda x: x.groups()[0][:3], regex=True)

    # delete commas after dates:
    dt['corrected'] = dt['corrected'].str.replace(r'(\d{1,2}),', r'\1', regex=True)
    # print(dt)

    # # Finding 2-digit year in line ends (after '/' or '-') converting them to 19XX
    # dt['extracted'] = dt['extracted'].str.replace(r'([/-])(\d{2})$', r'\g<1>19\g<2>', regex=True)
    # print(dt)

    # transforming all dates to unified format with pandas to_datetime()
    dt['date'] = pd.to_datetime(dt['corrected'], format='mixed')

    # xx years before 76 (2026 - 100/2) function to_datetime interpreted as 20xx, fixing:
    dt['date'] = dt['date'].apply(lambda x: x.replace(year=x.year - 100) if x.year > 2026 else x)

    # print(dt)

    # creating final order (Series):
    order = pd.Series(dt.sort_values(by='date', kind='stable').index)
    # kind='stable' TO KEEP THE ORIGINAL SEQUENCE IF SAME DATES!

    # print(order)
    return order  # Your answer here

# print(date_sorter())
# print('\n================================================================================\n')

# ------------------------- Method 2 -----------------------

def date_sorter2():
    import re
    order = None
    global dt2

    # 04/20/2009; 04/20/09; 4/20/09; 4/3/09; 4-13-89

    # Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
    # 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
    # Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
    # Feb 2009; Sep 2009; Oct 2010

    # 6/2008; 12/2009
    # 2009; 2010
    dt2 = pd.DataFrame()
    dt2['extracted'] = df2.str.extract(r'(\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})|(?:\d{1,2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,]*[ -]'
                    r'(?:\d{1,2}(?:st|nd|rd|th)?[, -]+)?\d{4}|(?:\d{1,2}/)?(?<!\d{3}-)(?<!\d)\d{4}(?!\d))')

    # replace weekdays with 3 letter abbrevations:
    dt2['corrected'] = dt2['extracted'].str.replace(
        r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,]*)',
        lambda x: x.groups()[0][:3], regex=True)

    # delete commas after dates:
    dt2['corrected'] = dt2['corrected'].str.replace(r'(\d{1,2}),', r'\1', regex=True)
    # print(dt2)

    # # Finding 2-digit year in line ends (after '/' or '-') converting them to 19XX
    # dt2['extracted'] = dt2['extracted'].str.replace(r'([/-])(\d{2})$', r'\g<1>19\g<2>', regex=True)
    # print(dt2)

    # transforming all dates to unified format with pandas to_datetime()
    dt2['date'] = pd.to_datetime(dt2['corrected'], format='mixed')

    # xx years before 76 (2026 - 100/2) function to_datetime interpreted as 20xx, fixing:
    dt2['date'] = dt2['date'].apply(lambda x: x.replace(year=x.year - 100) if x.year > 2026 else x)

    print(dt2)

    # creating final order (Series):
    order = pd.Series(dt2.sort_values(by='date', kind='stable').index)
    # kind='stable' TO KEEP THE ORIGINAL SEQUENCE IF SAME DATES!

    # print(order)
    return order  # Your answer here

# print(date_sorter2())


# ----------------------------------------------------------------------------------------------
#
# ------------- Проверочный Код -------------
#
# в варианте с одним regex разделенным | проверка идёт не по очереди по условиям трёх regex,
# а по символам строки, поэтому послденее выражение надо снабжать охранниками:
# (?<!\d{3}-)   : до 4 цифр года НЕТ трёх подряд цифр с дефисом (выключаем номера телефонов)
# (?<!\d)       : до 4 цифр года также нет цифр вплотную (чтобы не вырывал год из 1234546)
# \d{4}         : сами 4 цифры года
# (?!\d)        : чтобы не было цифр после.

line = '''.Spoke to sister Naomi Ely 708-810-7787 who reports he has been doing much better since he went to 
Dysart Clinic (he was drinking for a month leading up to this, his ammonia was high, and physicians were 
worried about early).  She feels his cognition is back to baseline, "100% better".  She says he has been 
successful in abstaining from substances as far as she knows, thinks a schedule is useful to him, 
doctor's appts etc.  Notes that he returned from LA in August 2008, gets bouts of "exhaustion" 
even in sobriety.  She denies ever witnessing any periods of manic behavior from patient.  
Their father has dementia that started at age 84.  Notes patient is living with uncle in Black River Falls 
(uncle is 89), lived with sister 3 months who also takes care of her own father in Talladega.  
She knows he is working on getting social security, subsidizing housing.  Stable situation with patient's 
girlfriend Nutt.Suicidal Behavior Hx of Suicidal Behavior: No'''

ps = pd.Series(line)

print('Проверка regex на предмет вытаскивания August 2008, который следует за телефонным номером: ',
      ps.str.extract(
        r'(\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})|'
        r'(?:\d{1,2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,]*[ -]'
        r'(?:\d{1,2}(?:st|nd|rd|th)?[, -]+)?\d{4}|(?:\d{1,2}/)?(?<!\d{3}-)(?<!\d)\d{4}(?!\d))'))

# ----------------------------------------------------------------------------------------------
#
# Consistency check:
#
# print(date_sorter2() == date_sorter())
#

# ========================= СОБИРАЕМ МЕДИЦИНСКИЙ ЖУРНАЛ ПО ДАТАМ ========================
#

order = date_sorter2()

# 1. Приписываем исходный текст из Series (df) в новую колонку таблицы dt2
# Так как индексы у df и dt2 изначально одинаковые, Pandas склеит их идеально строка к строке
dt2['text'] = df2

# 2. Сортируем ВЕСЬ датафрейм dt2 по колонке с датами
# Используем kind='stable', чтобы сохранить исходный порядок строк при одинаковых датах
dt2_sorted = dt2.sort_values(by='date', kind='stable')

# 3. Сбрасываем индексы (необязательно, но полезно для красоты)
# Чтобы новые строки шли по порядку от 0 до 499, а старые индексы сохранились в отдельной колонке
dt2_sorted = dt2_sorted.reset_index()

from pprint import pprint
dt2_sorted['date'] = dt2_sorted['date'].astype(str)
# Превращаем отсортированную таблицу в список словарей и красиво печатаем с ограничением ширины в 100 символов
pprint(dt2_sorted[['date', 'text']].to_dict('records'), width=200, sort_dicts=False)

#
# ------------------------- In Russian -------------------------

doc_r = []
with open('assets/dates_r.txt', encoding='utf-16') as file:
    # print(file.read())
    # file.seek(0)
    for line in file:
        doc_r.append(line)        # line.strip('\n') - if we want DF to be identical to SR

df_r = pd.Series(doc_r)

order = date_sorter2()

# 1. Приписываем исходный текст из Series (df) в новую колонку таблицы dt2
# Так как индексы у df и dt2 изначально одинаковые, Pandas склеит их идеально строка к строке
dt2['text'] = df_r

# 2. Сортируем ВЕСЬ датафрейм dt2 по колонке с датами
# Используем kind='stable', чтобы сохранить исходный порядок строк при одинаковых датах
dt2_sorted = dt2.sort_values(by='date', kind='stable')

# 3. Сбрасываем индексы (необязательно, но полезно для красоты)
# Чтобы новые строки шли по порядку от 0 до 499, а старые индексы сохранились в отдельной колонке
dt2_sorted = dt2_sorted.reset_index()

from pprint import pprint
dt2_sorted['date'] = dt2_sorted['date'].astype(str)
# Превращаем отсортированную таблицу в список словарей и красиво печатаем с ограничением ширины в 100 символов
pprint(dt2_sorted[['date', 'text']].to_dict('records'), width=200, sort_dicts=False)


#
#











