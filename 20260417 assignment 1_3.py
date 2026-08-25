import pandas as pd
import numpy as np

# Filter all warnings. If you would like to see the warnings, please comment the two lines below.
import warnings
warnings.filterwarnings('ignore')

# Снимаем ограничение на количество столбцов
pd.set_option('display.max_columns', 25)
# Увеличиваем общую ширину вывода (чтобы столбцы не переносились на новую строку)
pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows')

# Question 1:

Energy=pd.read_excel("Energy Indicators.xls", header=None)
Energy = Energy.iloc[18:245, 2:6].copy().reset_index(drop=True)

labels = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
Energy.columns=labels

cols =['Energy Supply', 'Energy Supply per Capita', '% Renewable']
Energy[cols] = Energy[cols].apply(pd.to_numeric, errors='coerce')
Energy['Energy Supply'] = Energy['Energy Supply'] * 1_000_000

Energy['Country'] = Energy['Country'].str.replace(r'\(.*\)' , '', regex=True)
Energy['Country'] = Energy['Country'].str.strip()
Energy['Country'] = Energy['Country'].str.replace(r'\d+$', '', regex=True)

rename_countries = {"Republic of Korea": "South Korea", "United States of America": "United States",
                    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                    "China, Hong Kong Special Administrative Region": "Hong Kong"}
Energy['Country'] = Energy['Country'].replace(rename_countries)

# =============================================================================================
GDP = pd.read_csv("world_bank.csv", skiprows=4)

rename_countries1 = {"Korea, Rep.": "South Korea", "Iran, Islamic Rep.": "Iran", "Hong Kong SAR, China": "Hong Kong"}
GDP['Country Name'] = GDP['Country Name'].replace(rename_countries1)
GDP.rename(columns={'Country Name':'Country'},inplace=True)

# =============================================================================================
ScimEn=pd.read_excel("scimagojr-3.xlsx")
# Этого делать НЕ НАДО, иначе Q2 неверный, но вообще там есть 2 страны с пробелами на конце:
# ScimEn['Country'] = ScimEn['Country'].str.strip()

# =============================================================================================
print(Energy)
print('=======================================================================')
print(GDP)
print('=======================================================================')
print(ScimEn)
print('=======================================================================')

# For Question 2 BEFORE Indexing:
#
# Количество строк при объединении всех данных (Outer)
# union_len = len(pd.merge(pd.merge(Energy, GDP, on='Country', how='outer'), ScimEn, on='Country', how='outer'))
# но лучше так:
union_len = len(pd.concat([Energy['Country'], GDP['Country'], ScimEn['Country']]).unique())
# Количество строк, которые попали в итоговую таблицу до обрезки (Inner)
intersect_len = len(pd.merge(pd.merge(Energy, GDP, on='Country'), ScimEn, on='Country'))
#
# Indexing to Merge:
Energy.set_index("Country", inplace=True)
GDP.set_index("Country", inplace=True)
ScimEn.set_index("Country", inplace=True)

# CHECK:
print('CHECK Energy for replaced Countries (Must be emplty):')
country_list = ['Australia', 'Bolivia', 'China', 'Hong Kong', 'China, Macao Special Administrative Region', 'Denmark', 'Falkland Islands', 'France', 'Greenland', 'Indonesia', 'Iran', 'Italy', 'Japan', 'Kuwait', 'Micronesia', 'Netherlands', 'Portugal', 'South Korea', 'Saudi Arabia', 'Serbia', 'Sint Maarten', 'Spain', 'Switzerland', 'Ukraine', 'United Kingdom', 'United States', 'Venezuela']

for i in range(len(country_list)):
    if country_list[i] not in Energy.index.values.tolist():
        print(country_list[i])
#If Country is not in the index, replace index with ['Country] in the above line.

# DF MERGE:
# (Можно было объединить по on='Country'):
# pd.merge(pd.merge(ScimEn, Energy, on='Country'), GDP.iloc[:, -10:], on='Country'
# но желать это надо ДО индексирования 'Country'
#
Sum_df = pd.merge(ScimEn, Energy, left_index=True, right_index=True, how='inner')
Sum_df = pd.merge(Sum_df, GDP.iloc[:, -10:], left_index=True, right_index=True, how='inner')

print('=======================================================================')
print(Sum_df)

# Question 2:
print('lenghs:')
print(len(Energy.index))
print(len(GDP.index))
print(len(ScimEn.index))
print(len(Sum_df.index))

print("union_len =", union_len)
print("intersect_len =", intersect_len)
print("Final Answer Q2 (Excluded Records):", union_len - intersect_len)
print('=======================================================================')

# Cutting to Top 15 Countries:
Sum_df = Sum_df.sort_values('Rank')[:15].copy()
print('FINAL TOP 15:')
print(Sum_df)

# Question 3:
#
# здесь надо использовать Sum_df, но это намного проще, я сделал из исходных таблиц
# GDP_Countries_Only = pd.merge(GDP, Energy[[]], left_index=True, right_index=True, how='inner')
# avgGDP = GDP_Countries_Only.apply(lambda x: np.mean(x.iloc[-10:]), axis=1).sort_values(ascending=False)[:15]
# print(GDP_Countries_Only.head(15))
#
# А вот так это в Q3:
avgGDP = Sum_df.apply(lambda x: np.mean(x.iloc[-10:]), axis=1).sort_values(ascending=False)
print(avgGDP)

# Question 4:

Sixth_GDP = Sum_df.loc[avgGDP.index[5]]
print(Sixth_GDP.tail(10))
Sixth_GDP_Change_10y = np.abs(Sixth_GDP.iloc[-10] - Sixth_GDP.iloc[-1])
print(Sixth_GDP_Change_10y)
print('check', 2666333396477.129883-2419630700401.72998)

# Question 5:

en_per_cap = np.nanmean(Sum_df['Energy Supply per Capita'])
print(en_per_cap)

# Question 6:

max_renw = (Sum_df['% Renewable'].idxmax(), np.nanmax(Sum_df['% Renewable']))
print(max_renw)

# Question 7:

Sum_df['CitRatio'] = Sum_df['Self-citations'] / Sum_df['Citations']
max_cit = (Sum_df['CitRatio'].idxmax(), np.nanmax(Sum_df['CitRatio']))
print(Sum_df)
print(max_cit)

# Question 8:

Sum_df['PopEst'] = Sum_df['Energy Supply'] / Sum_df['Energy Supply per Capita']
sorted_pop = Sum_df['PopEst'].sort_values(ascending=False)
maxpop_country = sorted_pop.index[2]
print(Sum_df)
print(maxpop_country)

# Question 9:

def plot9():
    import matplotlib.pyplot as plt
    Top15 = Sum_df
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])
    plt.show()

Sum_df['Citable docs per Capita'] = Sum_df['Citable documents'] / Sum_df['PopEst']
correlation = Sum_df['Citable docs per Capita'].corr(Sum_df['Energy Supply per Capita'])
print(Sum_df)
print(correlation)

plot9()

# Question 10:

rmed = Sum_df['% Renewable'].median()
Sum_df['Renew_Median'] = (Sum_df['% Renewable'] >= rmed).astype(int)
Renew_Median = Sum_df['Renew_Median']

print(Renew_Median)
# print(Sum_df)
print(rmed)

# Question 11:

ContinentDict = {'China': 'Asia',
                 'United States': 'North America',
                 'Japan': 'Asia',
                 'United Kingdom': 'Europe',
                 'Russian Federation': 'Europe',
                 'Canada': 'North America',
                 'Germany': 'Europe',
                 'India': 'Asia',
                 'France': 'Europe',
                 'South Korea': 'Asia',
                 'Italy': 'Europe',
                 'Spain': 'Europe',
                 'Iran': 'Asia',
                 'Australia': 'Australia',
                 'Brazil': 'South America'}

Sum_df['Continent'] = Sum_df.index.map(ContinentDict)
print(Sum_df)

# Тоже работает, но иерархическая шапка и названия столбцов другие:
# cont = Sum_df.groupby('Continent').agg({'Continent': np.size, 'PopEst': (np.nansum, np.nanmean, np.nanstd)})
# cont.columns = ['size', 'sum', 'mean', 'std']
cont = Sum_df.groupby('Continent')['PopEst'].agg(['size', 'sum', 'mean', 'std'])
print(cont)

# Question 12:

Sum_df['bins'] = pd.cut(Sum_df['% Renewable'], bins=5)
print(Sum_df)
Sum_df6 =  Sum_df.groupby(['Continent', 'bins']).size()
Sum_df6.index.names = ['Continent', '% Renewable']
print(Sum_df6)
print(len(Sum_df6))
print(Sum_df6.index)

# Question 13:

PopEst = Sum_df['PopEst'].apply(lambda x: '{:,}'.format(x))
print(PopEst)

# Optional 14:

def plot_optional():
    import matplotlib.pyplot as plt
    Top15 = Sum_df.copy()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter',
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'],
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'].iloc[i], Top15['% Renewable'].iloc[i]], ha='center')

    print("""    This is an example of a visualization that can be created to help understand the data.
    This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries'
    2014 GDP, and the color corresponds to the continent.""")
    plt.show()

plot_optional()

#
#
#


