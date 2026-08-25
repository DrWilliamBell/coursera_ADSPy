import pandas as pd
import numpy as np
import matplotlib as mpl
# mpl.use('Qt5Agg')                   # Use interactive Qt5
import matplotlib.pyplot as plt

# Снимаем ограничение на количество столбцов
pd.set_option('display.max_columns', 25)
# Увеличиваем общую ширину вывода (чтобы столбцы не переносились на новую строку)
pd.set_option('display.width', 500)
# pd.set_option('display.max_rows', None)
# Возврат:
# pd.reset_option('display.max_rows')

df = pd.read_csv('assets/4312418.csv')

#  I'll be using the folium package to render the data into a map in Jupyter.
import folium
from IPython.display import display

# get longitude and lattitude to plot
lons = df[['LATITUDE', 'LONGITUDE']].drop_duplicates()['LONGITUDE'].tolist()
lats = df[['LATITUDE', 'LONGITUDE']].drop_duplicates()['LATITUDE'].tolist()

# plot on a beautiful folium map
my_map = folium.Map(location = [lats[0], lons[0]], height = 900,  zoom_start = 9)
for lat, lon in zip(lats, lons):
    folium.Marker([lat, lon]).add_to(my_map)

# ДЛЯ PYCHARM:
my_map.save("map2_2.html")
# Чтобы браузер открылся сам:
import webbrowser
webbrowser.open("map2_2.html")

df = pd.read_csv('assets/4312418.csv')
print(df.head())

df['DATE'] = pd.to_datetime(df['DATE'])
df = df[~((df['DATE'].dt.month == 2) & (df['DATE'].dt.day == 29))]  # Dropping 29.02

dmax = df.groupby('DATE')[['TMAX']].max()
dmin = df.groupby('DATE')[['TMIN']].min()
# При использовании [] -> Series, [[]] -> DataFrame!
print(dmax.head())
print(dmin.head())
print(type(dmax), len(dmax))
print(type(dmin), len(dmin))

dmax.rename(columns={'TMAX': 'Data_Value'}, inplace=True)
dmin.rename(columns={'TMIN': 'Data_Value'}, inplace=True)

dmax514 = dmax[((dmax.index.year >= 2005) & (dmax.index.year <= 2014))]
dmin514 = dmin[((dmin.index.year >= 2005) & (dmin.index.year <= 2014))]
dmax15 = dmax[dmax.index.year == 2015]
dmin15 = dmin[dmin.index.year == 2015]
# index = для работы с Series (если не стояло as_index=False)
# пример: df.groupby('Date', as_index=False)['Data_Value'].max()
# Но и в DF при groupby колонка Date тоже стала индексом и изменить это можно через:
# dmax15 = dmax15.reset_index()

dmax514.index = dmax514.index.strftime('%m-%d')
dmin514.index = dmin514.index.strftime('%m-%d')
dmax15.index = dmax15.index.strftime('%m-%d')
dmin15.index = dmin15.index.strftime('%m-%d')

dmax514 = dmax514.groupby(dmax514.index)[['Data_Value']].max()
dmin514 = dmin514.groupby(dmin514.index)[['Data_Value']].min()
print(dmax514)
print(dmin514)
print(dmax15)
print(dmin15)

# ========================= PLOTTING ==========================
from calendar import month_abbr

plt.figure(figsize=(14,8))
plt.title('Temperature Extremes in London Area, England\n(2005-2014 vs 2015)', fontsize=16)
plt.xlabel('Day of the Year', fontsize=12, labelpad=13)
plt.ylabel('Temperature (C)', fontsize=12)
plt.margins(x=0, y=0.1)

plt.plot(dmax514['Data_Value'], linewidth=0.5, color='red', label='Max daily temp 2005-2014')
plt.plot(dmin514['Data_Value'], linewidth=0.5, color='blue', label='Min daily temp 2005-2014')

dmin15_ext = dmin15[dmin15['Data_Value'] < dmin514['Data_Value']]
dmax15_ext = dmax15[dmax15['Data_Value'] > dmax514['Data_Value']]
plt.scatter(dmin15_ext.index, dmin15_ext['Data_Value'], s=8, color='black', label='2015 daily Extremes greater/less than 2005-2014')
plt.scatter(dmax15_ext.index, dmax15_ext['Data_Value'], s=8, color='black')

plt.fill_between(range(365), dmin514['Data_Value'], dmax514['Data_Value'], facecolor='lightslategrey', alpha=0.2)
plt.legend(loc=4, frameon=False)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks(range(0, 360, 30), [])
# Ставим НАЗВАНИЯ месяцев со смещением 15 (в центр)
# Используем minor-тики (второстепенные деления) для текста
ax = plt.gca()
ax.set_xticks(range(15, 365, 30), minor=True)
ax.set_xticklabels(list(month_abbr)[1:], minor=True)
# Убираем короткие черточки у названий, чтобы они не двоились
ax.tick_params(axis='x', which='minor', size=0, pad=10)

plt.show()
plt.close()

# То же самое начисто с англ. комментариями для Coursera:
# ==========================================================================
# ============================= LONDON AREA ================================
# ==========================================================================
import pandas as pd
import numpy as np
import matplotlib as mpl
# mpl.use('Qt5Agg')                   # Use interactive Qt5
import matplotlib.pyplot as plt

# The London Area data file was requested from NOOA:
df = pd.read_csv('assets/4312418.csv')

# Dropping 29.02:
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[~((df['DATE'].dt.month == 2) & (df['DATE'].dt.day == 29))]

# Grouping Temp by dates ([[]] results DataFrames):
dmax = df.groupby('DATE')[['TMAX']].max()
dmin = df.groupby('DATE')[['TMIN']].min()

dmax.rename(columns={'TMAX': 'Data_Value'}, inplace=True)
dmin.rename(columns={'TMIN': 'Data_Value'}, inplace=True)

# Splitting 2005-2014 and 2015
dmax514 = dmax[((dmax.index.year >= 2005) & (dmax.index.year <= 2014))]
dmin514 = dmin[((dmin.index.year >= 2005) & (dmin.index.year <= 2014))]
dmax15 = dmax[dmax.index.year == 2015]
dmin15 = dmin[dmin.index.year == 2015]

# Leaving MM-DD (Cutting Year Value)
dmax514.index = dmax514.index.strftime('%m-%d')
dmin514.index = dmin514.index.strftime('%m-%d')
dmax15.index = dmax15.index.strftime('%m-%d')
dmin15.index = dmin15.index.strftime('%m-%d')

# Grouping 2005-2014 by Day of the Year:
dmax514 = dmax514.groupby(dmax514.index)[['Data_Value']].max()
dmin514 = dmin514.groupby(dmin514.index)[['Data_Value']].min()

# ========================= PLOTTING ==========================
from calendar import month_abbr

plt.figure(figsize=(14,8))
plt.title('Temperature Extremes in London Area, England\n(2005-2014 vs 2015)', fontsize=16)
plt.xlabel('Day of the Year', fontsize=12, labelpad=13)
plt.ylabel('Temperature (C)', fontsize=12)
plt.margins(x=0, y=0.1)

# 2005-2014 Plotting:
plt.plot(dmax514['Data_Value'], linewidth=0.5, color='red', label='Max daily temp 2005-2014')
plt.plot(dmin514['Data_Value'], linewidth=0.5, color='blue', label='Min daily temp 2005-2014')

# Creating DFs with 2015 extremes over 2005-2014 and scattering them:
dmin15_ext = dmin15[dmin15['Data_Value'] < dmin514['Data_Value']]
dmax15_ext = dmax15[dmax15['Data_Value'] > dmax514['Data_Value']]

plt.scatter(dmin15_ext.index, dmin15_ext['Data_Value'], s=8, color='black', label='2015 daily Extremes greater/less than 2005-2014')
plt.scatter(dmax15_ext.index, dmax15_ext['Data_Value'], s=8, color='black')

# Shading in between 2005-2014:
plt.fill_between(range(365), dmin514['Data_Value'], dmax514['Data_Value'], facecolor='lightslategrey', alpha=0.2)

# Legend (labels in plottings):
plt.legend(loc=4, frameon=False)

# Cleaning:
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Xticks and Months on X axis
# (minor ticks temporarily used to shift Months on 15 pix):
plt.xticks(range(0, 360, 30), [])
ax = plt.gca()
ax.set_xticks(range(15, 365, 30), minor=True)
ax.set_xticklabels(list(month_abbr)[1:], minor=True)
# Cleaning minors and moving X Label down 10 pix:
ax.tick_params(axis='x', which='minor', size=0, pad=10)

# plt.savefig('ANV_Assignment2_plot.png', dpi=300)
plt.show()
plt.close()

