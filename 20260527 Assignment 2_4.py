import pandas as pd

# disable output limits
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ======================= Datasets ======================
# https://www.nomisweb.co.uk/query/select/getdatasetbytheme.asp?theme=78
# https://data.london.gov.uk/dataset/?org=409911d3-83b3-43f6-8374-53a2aea313ee&masthead=ons&geo=Local%20Authority&topics=demographics&q=census%202001%20ethnic
# http://webarchive.nationalarchives.gov.uk/20160105160709/http://www.ons.gov.uk/ons/rel/census/2011-census/key-statistics-for-local-authorities-in-england-and-wales/rft-table-ks201ew.xls
# https://data.london.gov.uk/dataset/ethnic-groups-by-borough-2n858/
# https://www.ons.gov.uk/datasets/TS021/editions/2021/versions/3
# 1981 census data from Wikipedia (direct numbers, not dataset):
# https://en.wikipedia.org/wiki/Ethnic_groups_in_London#cite_note-:362-13

# lines indexes (our ethnic groups)
index_labels = ['White', 'Asian', 'Black', 'Mixed', 'Other', 'Non-White Total', 'Total']

# create base dataframe
df_main = pd.DataFrame(index=index_labels)

# =================== 1981 (Estimations) ===================
# 1981 census data (manual input from Wiki, no totals)
df_main['1981'] = [5893973, 425426, 405394, 0, 80793, 0, 0]

# ======================= 1991 Census ======================
# download 1991 census data from .xls
df91 = pd.read_excel('Assignment 2_4/1991_ethnic.xls')
# cleaning data and turn 2nd column to float64 and NaN
df91.iloc[:, 1] = pd.to_numeric(df91.iloc[:, 1], errors='coerce')
# calculating aggregated numbers (by indexes at this stage, as the dataset is small)
df_main.loc['White', '1991'] = df91.iloc[6, 1]
df_main.loc['Black', '1991'] = df91.iloc[[7, 8, 9], 1].sum()
df_main.loc['Asian', '1991'] = df91.iloc[[10, 11, 12, 13, 14], 1].sum()
df_main.loc['Mixed', '1991'] = 0
df_main.loc['Other', '1991'] = df91.iloc[15, 1]

# function to calculate sums and totals
def update_totals(df, column):
    # calculate Non-White Total as sum of 4 groups
    df.loc['Non-White Total', column] = df.loc[['Asian', 'Black', 'Mixed', 'Other'], column].sum()
    # calculate Total
    df.loc['Total', column] = df.loc['White', column] + df.loc['Non-White Total', column]
    # converting to int
    df[column] = df[column].astype(int)

# apply totals calc function to 1981 and 1991
update_totals(df_main, '1981')
update_totals(df_main, '1991')

# ======================= 2001 Census ======================
# download 2001 census data from .csv
df01 = pd.read_csv('Assignment 2_4/2001_ethnic.csv')
# extracting London data line [41] and converting data to float64
london_totals = pd.to_numeric(df01.iloc[41], errors='coerce')
# cleaning data - remove the prefix
london_totals.index = london_totals.index.str.replace('People in ethnic groups - ', '', regex=False)

# aggregate data with regex (filter by keywords)
df_main.loc['White', '2001'] = london_totals.filter(regex='^White').sum()
df_main.loc['Black', '2001'] = london_totals.filter(regex='^Black').sum()
df_main.loc['Mixed', '2001'] = london_totals.filter(regex='^Mixed').sum()

# for Asian: aggregate all by keyword 'Asian' plus ending on 'Chinese'
asian_base = london_totals.filter(regex='^Asian').sum()
chinese_pure = london_totals.filter(regex='Chinese$').iloc[0]  # '$' means line end
df_main.loc['Asian', '2001'] = asian_base + chinese_pure
# Other: strictly ending on 'Other Ethnic Group'
df_main.loc['Other', '2001'] = london_totals.filter(regex='Other Ethnic Group$').iloc[0]
# calculating totals and converting to int
update_totals(df_main, '2001')

# ======================= 2011 Census ======================
# download 2001 census data from .xls sheet 'KS201EW_Numbers'
df11 = pd.read_excel('Assignment 2_4/2011_ethnic.xls', sheet_name='KS201EW_Numbers')
# converting line 9 into df columns names to have new Series lines indexes in place
df11.columns = df11.iloc[9]
# extracting London data line [269] and converting data to float64
london_totals = pd.to_numeric(df11.iloc[269], errors='coerce')

# aggregate data with regex (filter by keywords)
df_main.loc['White', '2011'] = london_totals.filter(regex='^White').sum()
df_main.loc['Black', '2011'] = london_totals.filter(regex='^Black').sum()
df_main.loc['Mixed', '2011'] = london_totals.filter(regex='^Mixed').sum()
df_main.loc['Asian', '2011'] = london_totals.filter(regex='^Asian').sum()
df_main.loc['Other', '2011'] = london_totals.filter(regex='^Other').sum()
# calculating totals and converting to int
update_totals(df_main, '2011')

# ======================= 2021 Census ======================
# download 2001 census data from .csv
df21 = pd.read_csv('Assignment 2_4/2021_ethnic.csv')
# selecting London lines only (codes starting from E09)
df21 = df21[df21['Lower tier local authorities Code'].str.startswith('E09', na=False)]
# converting 'Observation' data to float64
df21['Observation'] = pd.to_numeric(df21['Observation'], errors='coerce')
# grouping London local authorities by ethnic groups and summarize Observations
# (turning the result into Series, where index = Ethnic group)
london_totals = df21.groupby('Ethnic group (20 categories)')['Observation'].sum()

# aggregate data with regex (filter by keywords)
df_main.loc['White', '2021'] = london_totals.filter(regex='^White').sum()
df_main.loc['Black', '2021'] = london_totals.filter(regex='^Black').sum()
df_main.loc['Mixed', '2021'] = london_totals.filter(regex='^Mixed').sum()
df_main.loc['Asian', '2021'] = london_totals.filter(regex='^Asian').sum()
df_main.loc['Other', '2021'] = london_totals.filter(regex='^Other').sum()
# calculating totals and converting to int
update_totals(df_main, '2021')

# ===================== 2012-2020 Census ====================
# download 2012-2020 census data from .xls as whole (all sheets)
xls = pd.ExcelFile('Assignment 2_4/2011_2020_ethnic.xls')

# Looping each sheet
for sheet_name in xls.sheet_names:
    # IF sheet = 'Metadata' — skip it
    if sheet_name == 'Metadata':
        continue
    # sheet_name contains year ('2012', '2013'...)
    df_year = pd.read_excel(xls, sheet_name=sheet_name)

    # extract one line, where Area == 'London'
    # .iloc[0] converts it into Series
    london_totals = df_year[df_year['Area'] == 'London'].iloc[0]

    # write values to df_main directly by indexes [2, 3, 4, 5]
    df_main.loc['White', sheet_name] = pd.to_numeric(london_totals.iloc[2], errors='coerce')
    df_main.loc['Asian', sheet_name] = pd.to_numeric(london_totals.iloc[3], errors='coerce')
    df_main.loc['Black', sheet_name] = pd.to_numeric(london_totals.iloc[4], errors='coerce')
    df_main.loc['Mixed', sheet_name] = pd.to_numeric(london_totals.iloc[5], errors='coerce')
    df_main.loc['Other', sheet_name] = 0  # included into Mixed/Other - no separate value here

    # calculating totals and converting to int
    update_totals(df_main, sheet_name)

# sorting columns by years chronologically
df_main = df_main.reindex(columns=sorted(df_main.columns))

# ===================== Merging Mixed and Other ====================
# (because we do not have full data for all periods for these lines)
# create new line 'Mixed / Other', concatenating 'Mixed' and 'Other'
df_main.loc['Mixed / Other'] = df_main.loc['Mixed'] + df_main.loc['Other']
# dropping 'Mixed' and 'Other'
df_main = df_main.drop(['Mixed', 'Other'])
# new nice lineup
correct_order = ['White', 'Asian', 'Black', 'Mixed / Other', 'Non-White Total', 'Total']
df_main = df_main.reindex(correct_order)

print("\n=== Aggregated Absolute Values (Persons) ===")
print(df_main)

# df with percentage (no Totals)
df_percent = df_main.drop('Total')
df_percent = ((df_percent / df_main.loc['Total']) * 100).round(2)

print("\n=== Percentage Distribution (%) ===")
print(df_percent)


# ============================ Plotting ============================
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib style (option)
# plt.style.use('bmh')

# Seaborn style
sns.set_theme(style="white")
plt.figure(figsize=(11, 6))

# setting yaxis step = 10%
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(10))
# leaving the horizontal soft grid only to read %
plt.grid(visible=True, axis='y', linestyle=':', alpha=0.7)

# Transpose and drop 'Non-White Total' to avoid scale distortion
df_plot = df_percent.T
print(df_plot)

# !!! CRITICAL FOR PROPORTIONAL TIMELINE !!!
# Convert string years ('1981', '1991'...) into integers (1981, 1991...)
# This forces Matplotlib to treat the X-axis as a linear numeric scale,
# automatically generating mathematically correct spacing between points.
df_plot.index = df_plot.index.astype(int)

# Define logical, highly professional colors
color_map = {
    'White': '#7f8c8d',            # Neutral slate gray
    'Non-White Total': '#b33939',  # Deep elegant red (to highlight the mirror effect)
    'Asian': '#4a69bd',            # Deep blue
    'Black': '#2c2c54',            # Midnight dark
    'Mixed / Other': '#d35400'     # Terracotta orange
}

# Draw lines for each category using composite lines for sub-groups
for group in df_plot.columns:
    is_main = group in ['White', 'Non-White Total']

    if is_main:
        # Main lines are just thick, solid, and heavy
        plt.plot(
            df_plot.index,
            df_plot[group],
            linewidth=4.0,
            linestyle='-',
            color=color_map.get(group, '#333'),
            alpha=0.95,
            antialiased=True
        )
    else:
        # Subgroups: Composite "Inline" effect (Dark outer line + light inner line)
        base_color = color_map.get(group, '#333')

        # 1. Draw the outer line (darker base, 3px wide)
        plt.plot(
            df_plot.index,
            df_plot[group],
            linewidth=3.0,
            linestyle='-',
            color=base_color,
            alpha=0.85,
            antialiased=True
        )

        # 2. Draw the inner line directly on top (1px wide, white or very light gray)
        # Using white creates a beautiful "hollow" or striped effect inside the line
        plt.plot(
            df_plot.index,
            df_plot[group],
            linewidth=1.0,
            linestyle='-',
            color='#ffffff',  # Clean white center stripe
            alpha=0.95,
            antialiased=True
        )
    # === Direct Labeling at the End of the Lines ===
    # Get the last coordinate (Year 2021)
    last_year = df_plot.index[-1]
    last_value = df_plot.loc[last_year, group]

    # Define text and manual vertical shifts to fix overlapping
    if group == 'White':
        label_text = f"{group} ({last_value}%)"
        y_position = last_value
    elif group == 'Non-White Total':
        label_text = f"{group} ({last_value}%)"
        y_position = last_value
    else:
        # Add an arrow to show it belongs to Non-White Total
        # 'r' prefix signals raw string for LaTeX interpretation.
        # '$\hookrightarrow$' builds a beautiful smooth curved arrow that works on Windows!
        label_text = fr"   $\hookrightarrow$ {group} ({last_value}%)"

        # FIX OVERLAPPING: Manual micro-shifts for sub-groups in year 2021
        if group == 'Black':
            y_position = last_value + 1.5  # Shift Black slightly up
        elif group == 'Mixed / Other':
            y_position = last_value - 1.5  # Shift Mixed slightly down
        else:
            y_position = last_value  # Asian stays as is

    # Add text label slightly shifted to the right (+0.5 of X-axis)
    plt.text(
        last_year + 0.5,
        y_position,
        label_text,
        fontsize=10,
        fontweight='bold' if is_main else 'normal',
        color=color_map.get(group, '#333'),
        va='center'  # Vertically centered on the line end
    )

# Fine-tuning X-axis ticks (to show actual years without decimals)
# We strictly define ticks for all historical points present in our dataset
all_years = df_plot.index.tolist()
# Fine-tuning X-axis ticks (Odd years only)
visible_labels = [str(y) if y % 2 != 0 else '' for y in all_years]

# Apply custom labels and rotate them for professional look
plt.xticks(all_years, labels=visible_labels, rotation=45, fontsize=10)

# Expand X-axis limit slightly to the right to fit the end text labels comfortably
plt.xlim(all_years[0], all_years[-1] + 8)

# Chart configuration (Title left-aligned)
plt.title('The Demographic Mirror: Ethnic Trends in London (1981 - 2021)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=12, labelpad=10)
plt.ylabel('% of Total Population', fontsize=12, labelpad=10)
plt.ylim(0, 100)

# === FIXED SIDE-NOTE: Elegant layout matching the requested subtitle style ===
# Coordinates: X = 2021.5 (right after chart), Y = 2.0 (near bottom right)
# Styled strictly in muted dark gray, normal weight, italic, size 9.5
plt.text(
    2021.5, 80.0,
    "Note: Non-White Total =\nAsian + Black + Mixed / Other",
    fontsize=9.5,
    fontweight='normal',
    fontstyle='italic',
    color='#555555',
    va='bottom',
    ha='left'
)

# removing the graph frames
sns.despine(left=True, bottom=True)
plt.tight_layout()

# Saving the chart to PNG
# 'dpi=300' ensures high professional resolution for reports/presentations
# 'bbox_inches="tight"' prevents the legend on the right from being cut off
plt.savefig('Assignment 2_4/london_ethnic_1981_2021.png', dpi=300, bbox_inches='tight')

# Display the final chart
plt.show()


















