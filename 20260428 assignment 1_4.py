import pandas as pd
import numpy as np
import scipy.stats as stats
import re

# =============================================================================================
# Question 1:

def nhl_correlation():
    nhl_df = pd.read_csv("assets/nhl.csv")
    cities = pd.read_html("assets/wikipedia_data.html")[1]
    cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]

    nhl_df = nhl_df[nhl_df["year"] == 2018]
    nhl_df["team"] = nhl_df["team"].str.replace(r'\*$', '', regex=True)
    nhl_df = nhl_df[["team", "W", "L"]]
    nhl_df = nhl_df[~nhl_df['W'].str.contains("Division")].reset_index(drop=True)
    nhl_df["team_s"] = nhl_df["team"].str.split().str[-1]
    nhl_df['W'] = pd.to_numeric(nhl_df['W'])
    nhl_df['L'] = pd.to_numeric(nhl_df['L'])
    nhl_df["WL_ratio"] = (nhl_df["W"]) / (nhl_df["W"] + nhl_df["L"])

    for x in range(2, 6):
        cities.iloc[:, x] = cities.iloc[:, x].str.replace(r'\[.*\]', '', regex=True)
    cities.rename(columns={'Population (2016 est.)[8]': 'Population'}, inplace=True)
    cities['Population'] = cities['Population'].astype(float)

    for idx, i in nhl_df["team_s"].items():
        found = cities[cities["NHL"].str.contains(i)]
        if not found.empty:
            nhl_df.loc[idx, "MP_area"] = found.iloc[0, 0]

    print(nhl_df)

    nhl_wl = nhl_df.groupby('MP_area')['WL_ratio'].mean().reset_index()
    print(nhl_wl)

    final_df = pd.merge(nhl_wl, cities, left_on='MP_area', right_on='Metropolitan area')
    print(final_df)

    population_by_region = final_df['Population']   # pass in metropolitan area population from cities
    win_loss_by_region = final_df['WL_ratio']       # pass in win/loss ratio from nhl_df
                                                    # in the same order as cities["Metropolitan area"]
    print(population_by_region, win_loss_by_region)

    # raise NotImplementedError()

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"

    res, pval = stats.pearsonr(population_by_region, win_loss_by_region)
    return res

# print(nhl_correlation())

# =============================================================================================
# Question 2:

def nba_correlation():
    nba_df=pd.read_csv("assets/nba.csv")
    cities=pd.read_html("assets/wikipedia_data.html")[1]
    cities=cities.iloc[:-1,[0,3,5,6,7,8]]

    nba_df = nba_df[nba_df["year"] == 2018]
    nba_df = nba_df[["team", "W", "L"]]
    nba_df["team"] = nba_df["team"].str.replace(r'\(.*\)', '', regex=True)
    nba_df["team"] = nba_df["team"].str.strip()
    nba_df["team"] = nba_df["team"].str.replace(r'\*$', '', regex=True)
    nba_df["team_s"] = nba_df["team"].str.split().str[-1]
    nba_df['W'] = pd.to_numeric(nba_df['W'])
    nba_df['L'] = pd.to_numeric(nba_df['L'])
    nba_df["WL_ratio"] = (nba_df["W"]) / (nba_df["W"] + nba_df["L"])

    for x in range(2, 6):
        cities.iloc[:, x] = cities.iloc[:, x].str.replace(r'\[.*\]', '', regex=True)
    cities.rename(columns={'Population (2016 est.)[8]': 'Population'}, inplace=True)
    cities['Population'] = cities['Population'].astype(float)

    for idx, i in nba_df["team_s"].items():
        found = cities[cities["NBA"].str.contains(i)]
        if not found.empty:
            nba_df.loc[idx, "MP_area"] = found.iloc[0, 0]
            # print(idx, found.iloc[0, 0], i)
    print(nba_df)

    nba_wl = nba_df.groupby('MP_area')['WL_ratio'].mean().reset_index()
    print(nba_wl)

    final_df = pd.merge(nba_wl, cities, left_on='MP_area', right_on='Metropolitan area')
    print(final_df)

    # raise NotImplementedError()

    population_by_region = final_df['Population']   # pass in metropolitan area population from cities
    win_loss_by_region = final_df['WL_ratio']       # pass in win/loss ratio from nhl_df in the same order
                                                    # as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"

    print(population_by_region, win_loss_by_region)

    res, pval = stats.pearsonr(population_by_region, win_loss_by_region)
    return res

# print(nba_correlation())

# =============================================================================================
# Question 3:

def mlb_correlation():
    mlb_df=pd.read_csv("assets/mlb.csv")
    cities=pd.read_html("assets/wikipedia_data.html")[1]
    cities=cities.iloc[:-1,[0,3,5,6,7,8]]

    mlb_df = mlb_df[mlb_df["year"] == 2018]
    mlb_df = mlb_df[["team", "W", "L"]]
    mlb_df["team_s"] = mlb_df["team"].str.split().str[-1]
    mlb_df.iloc[0, 3] = "Red Sox"
    mlb_df.iloc[8, 3] = "White Sox"
    mlb_df['W'] = pd.to_numeric(mlb_df['W'])
    mlb_df['L'] = pd.to_numeric(mlb_df['L'])
    mlb_df["WL_ratio"] = (mlb_df["W"]) / (mlb_df["W"] + mlb_df["L"])

    for x in range(2, 6):
        cities.iloc[:, x] = cities.iloc[:, x].str.replace(r'\[.*\]', '', regex=True)
    cities.rename(columns={'Population (2016 est.)[8]': 'Population'}, inplace=True)
    cities['Population'] = cities['Population'].astype(float)

    for idx, i in mlb_df["team_s"].items():
        found = cities[cities["MLB"].str.contains(i)]
        if not found.empty:
            mlb_df.loc[idx, "MP_area"] = found.iloc[0, 0]
            # print(idx, found.iloc[0, 0], i)
    print(mlb_df)

    mlb_wl = mlb_df.groupby('MP_area')['WL_ratio'].mean().reset_index()
    print(mlb_wl)

    final_df = pd.merge(mlb_wl, cities, left_on='MP_area', right_on='Metropolitan area')
    print(final_df)

    # raise NotImplementedError()

    population_by_region = final_df['Population']   # pass in metropolitan area population from cities
    win_loss_by_region = final_df['WL_ratio']       # pass in win/loss ratio from nhl_df in the same order
                                                    # as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    print(population_by_region, win_loss_by_region)

    res, pval = stats.pearsonr(population_by_region, win_loss_by_region)
    return res

# print(mlb_correlation())

# =============================================================================================
# Question 4:

def nfl_correlation():
    nfl_df=pd.read_csv("assets/nfl.csv")
    cities=pd.read_html("assets/wikipedia_data.html")[1]
    cities=cities.iloc[:-1,[0,3,5,6,7,8]]

    nfl_df = nfl_df[nfl_df["year"] == 2018]
    nfl_df = nfl_df[["team", "W", "L"]]
    nfl_df["team"] = nfl_df["team"].str.replace(r'\*$', '', regex=True)
    nfl_df["team"] = nfl_df["team"].str.replace(r'\+$', '', regex=True)
    nfl_df = nfl_df[~nfl_df['W'].str.contains("AFC")].reset_index(drop=True)
    nfl_df = nfl_df[~nfl_df['W'].str.contains("NFC")].reset_index(drop=True)
    nfl_df["team_s"] = nfl_df["team"].str.split().str[-1]
    nfl_df['W'] = pd.to_numeric(nfl_df['W'])
    nfl_df['L'] = pd.to_numeric(nfl_df['L'])
    nfl_df["WL_ratio"] = (nfl_df["W"]) / (nfl_df["W"] + nfl_df["L"])

    for x in range(2, 6):
        cities.iloc[:, x] = cities.iloc[:, x].str.replace(r'\[.*\]', '', regex=True)
    cities.rename(columns={'Population (2016 est.)[8]': 'Population'}, inplace=True)
    cities['Population'] = cities['Population'].astype(float)

    for idx, i in nfl_df["team_s"].items():
        found = cities[cities["NFL"].str.contains(i)]
        if not found.empty:
            nfl_df.loc[idx, "MP_area"] = found.iloc[0, 0]
    print(nfl_df)

    nfl_wl = nfl_df.groupby('MP_area')['WL_ratio'].mean().reset_index()
    print(nfl_wl)

    final_df = pd.merge(nfl_wl, cities, left_on='MP_area', right_on='Metropolitan area')
    print(final_df)

    # raise NotImplementedError()

    population_by_region = final_df['Population']   # pass in metropolitan area population from cities
    win_loss_by_region = final_df['WL_ratio']       # pass in win/loss ratio from nhl_df in the same order
                                                    # as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q3: There should be 29 teams being analysed for nfl"

    print(population_by_region, win_loss_by_region)

    res, pval = stats.pearsonr(population_by_region, win_loss_by_region)
    return res

# print(nfl_correlation())

# =============================================================================================
# Question 5:

def sports_team_performance():
    mlb_df = pd.read_csv("assets/mlb.csv")
    nhl_df = pd.read_csv("assets/nhl.csv")
    nba_df = pd.read_csv("assets/nba.csv")
    nfl_df = pd.read_csv("assets/nfl.csv")
    cities = pd.read_html("assets/wikipedia_data.html")[1]
    cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]
# Cities Preparation
    for x in range(2, 6):
        cities.iloc[:, x] = cities.iloc[:, x].str.replace(r'\[.*\]', '', regex=True)
    cities.rename(columns={'Population (2016 est.)[8]': 'Population'}, inplace=True)
    cities['Population'] = cities['Population'].astype(float)
# NHL
    nhl_df = nhl_df[nhl_df["year"] == 2018]
    nhl_df["team"] = nhl_df["team"].str.replace(r'\*$', '', regex=True)
    nhl_df = nhl_df[["team", "W", "L"]]
    nhl_df = nhl_df[~nhl_df['W'].str.contains("Division")].reset_index(drop=True)
    nhl_df["team_s"] = nhl_df["team"].str.split().str[-1]
    nhl_df['W'] = pd.to_numeric(nhl_df['W'])
    nhl_df['L'] = pd.to_numeric(nhl_df['L'])
    nhl_df["WL_ratio"] = (nhl_df["W"]) / (nhl_df["W"] + nhl_df["L"])

    for idx, i in nhl_df["team_s"].items():
        found = cities[cities["NHL"].str.contains(i)]
        if not found.empty:
            nhl_df.loc[idx, "MP_area"] = found.iloc[0, 0]
    nhl_wl = nhl_df.groupby('MP_area')['WL_ratio'].mean().reset_index()
# NBA
    nba_df = nba_df[nba_df["year"] == 2018]
    nba_df = nba_df[["team", "W", "L"]]
    nba_df["team"] = nba_df["team"].str.replace(r'\(.*\)', '', regex=True)
    nba_df["team"] = nba_df["team"].str.strip()
    nba_df["team"] = nba_df["team"].str.replace(r'\*$', '', regex=True)
    nba_df["team_s"] = nba_df["team"].str.split().str[-1]
    nba_df['W'] = pd.to_numeric(nba_df['W'])
    nba_df['L'] = pd.to_numeric(nba_df['L'])
    nba_df["WL_ratio"] = (nba_df["W"]) / (nba_df["W"] + nba_df["L"])

    for idx, i in nba_df["team_s"].items():
        found = cities[cities["NBA"].str.contains(i)]
        if not found.empty:
            nba_df.loc[idx, "MP_area"] = found.iloc[0, 0]
    nba_wl = nba_df.groupby('MP_area')['WL_ratio'].mean().reset_index()
# MLB
    mlb_df = mlb_df[mlb_df["year"] == 2018]
    mlb_df = mlb_df[["team", "W", "L"]]
    mlb_df["team_s"] = mlb_df["team"].str.split().str[-1]
    mlb_df.iloc[0, 3] = "Red Sox"
    mlb_df.iloc[8, 3] = "White Sox"
    mlb_df['W'] = pd.to_numeric(mlb_df['W'])
    mlb_df['L'] = pd.to_numeric(mlb_df['L'])
    mlb_df["WL_ratio"] = (mlb_df["W"]) / (mlb_df["W"] + mlb_df["L"])

    for idx, i in mlb_df["team_s"].items():
        found = cities[cities["MLB"].str.contains(i)]
        if not found.empty:
            mlb_df.loc[idx, "MP_area"] = found.iloc[0, 0]
    mlb_wl = mlb_df.groupby('MP_area')['WL_ratio'].mean().reset_index()
# NFL
    nfl_df = nfl_df[nfl_df["year"] == 2018]
    nfl_df = nfl_df[["team", "W", "L"]]
    nfl_df["team"] = nfl_df["team"].str.replace(r'\*$', '', regex=True)
    nfl_df["team"] = nfl_df["team"].str.replace(r'\+$', '', regex=True)
    nfl_df = nfl_df[~nfl_df['W'].str.contains("AFC")].reset_index(drop=True)
    nfl_df = nfl_df[~nfl_df['W'].str.contains("NFC")].reset_index(drop=True)
    nfl_df["team_s"] = nfl_df["team"].str.split().str[-1]
    nfl_df['W'] = pd.to_numeric(nfl_df['W'])
    nfl_df['L'] = pd.to_numeric(nfl_df['L'])
    nfl_df["WL_ratio"] = (nfl_df["W"]) / (nfl_df["W"] + nfl_df["L"])

    for idx, i in nfl_df["team_s"].items():
        found = cities[cities["NFL"].str.contains(i)]
        if not found.empty:
            nfl_df.loc[idx, "MP_area"] = found.iloc[0, 0]
    nfl_wl = nfl_df.groupby('MP_area')['WL_ratio'].mean().reset_index()
#
    sports = ['NFL', 'NBA', 'NHL', 'MLB']
    p_values = pd.DataFrame({k: np.nan for k in sports}, index=sports)
    spdict = {'NFL': nfl_wl,
              'NBA': nba_wl,
              'NHL': nhl_wl,
              'MLB': mlb_wl}
    for a in sports:
        for b in sports:
            if a != b:
                df_a = spdict[a]
                df_b = spdict[b]
                merged = pd.merge(df_a, df_b, on='MP_area', how='inner')
                teststat, pval = stats.ttest_rel(merged['WL_ratio_x'], merged['WL_ratio_y'])
                p_values.loc[a, b] = pval
            else:
                p_values.loc[a, b] = np.nan

    # raise NotImplementedError()

    # Note: p_values is a full dataframe, so df.loc["NFL","NBA"] should be the same as df.loc["NBA","NFL"] and
    # df.loc["NFL","NFL"] should return np.nan
    #
    assert abs(p_values.loc["NBA", "NHL"] - 0.02) <= 1e-2, "The NBA-NHL p-value should be around 0.02"
    assert abs(p_values.loc["MLB", "NFL"] - 0.80) <= 1e-2, "The MLB-NFL p-value should be around 0.80"
    return p_values

print(sports_team_performance())

#
























