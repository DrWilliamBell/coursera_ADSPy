# Question 1
# def proportion_of_education():
#     import pandas as pd
#     df = pd.read_csv('C:/Python/Jupyter/coursera/course1/NISPUF17.csv')
#
#     proportions = {"less than high school": None,
#                    "high school": None,
#                    "more than high school but not college": None,
#                    "college": None}
#
#     total = len(df)
#     param = 0
#     for i in proportions:
#         param += 1
#         proportions[i] = len(df[df['EDUC1'] == param]) / total
#     return proportions
#
#     raise NotImplementedError()
#
# assert type(proportion_of_education())==type({}), "You must return a dictionary."
# assert len(proportion_of_education()) == 4, "You have not returned a dictionary with four items in it."
# assert "less than high school" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
# assert "high school" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
# assert "more than high school but not college" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
# assert "college" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
#
# print(proportion_of_education())

# Question 2

# def average_influenza_doses():
#     import pandas as pd
#     df = pd.read_csv('C:/Python/Jupyter/coursera/course1/NISPUF17.csv')
#     df1 = df[['CBF_01','P_NUMFLU']].dropna()
#     # # print(df1.head(60))
#     # # yes = df1[df1['CBF_01'] == 1]['P_NUMFLU']
#     # # no = df1[df1['CBF_01'] == 2]['P_NUMFLU']
#     # # print(sum(yes), len(yes), sum(yes)/len(yes))
#     # # m_yes = yes.mean()
#     # # print(sum(no), len(no), sum(no) / len(no))
#     # # m_no = no.mean()
#     # # m_yes = df1[df1['CBF_01'] == 1].mean()
#     # # m_no = df1[df1['CBF_01'] == 2].mean()
#     # # print(m_yes.iloc[1], m_no.iloc[1])
#
#     m_yes = df1[df1['CBF_01'] == 1]['P_NUMFLU'].mean()
#     m_no = df1[df1['CBF_01'] == 2]['P_NUMFLU'].mean()
#     return (m_yes, m_no)
#
#     raise NotImplementedError()
#
# assert len(average_influenza_doses())==2, "Return two values in a tuple, the first for yes and the second for no."
#
# print(average_influenza_doses())

# Question 3

# def chickenpox_by_sex():
#     import pandas as pd
#     df = pd.read_csv('C:/Python/Jupyter/coursera/course1/NISPUF17.csv')
#     df1 = df[['SEX', 'HAD_CPOX', 'P_NUMVRC']].dropna()
#     print(df1.head(30))
#
#     ratios = {"male": None,
#               "female": None}
#     boys = df1[df1['SEX'] == 1]
#     girls = df1[df1['SEX'] == 2]
#
#     ratios['male'] = len(boys[boys['HAD_CPOX'].eq(1) & boys['P_NUMVRC'].ge(1)])/len(boys[boys['HAD_CPOX'].eq(2) & boys['P_NUMVRC'].ge(1)])
#     ratios['female'] = len(girls[girls['HAD_CPOX'].eq(1) & girls['P_NUMVRC'].ge(1)])/len(girls[girls['HAD_CPOX'].eq(2) & girls['P_NUMVRC'].ge(1)])
#
#     return ratios
#
#     raise NotImplementedError()
#
# assert len(chickenpox_by_sex())==2, "Return a dictionary with two items, the first for males and the second for females."
#
# print(chickenpox_by_sex())

# Question 4

def corr_chickenpox():
    import scipy.stats as stats
    import numpy as np
    import pandas as pd

    # this is just an example dataframe
    df = pd.DataFrame({"had_chickenpox_column": np.random.randint(1, 3, size=(100)),
                       "num_chickenpox_vaccine_column": np.random.randint(0, 6, size=(100))})

    # here is some stub code to actually run the correlation
    corr, pval = stats.pearsonr(df["had_chickenpox_column"], df["num_chickenpox_vaccine_column"])

    print(df.head(10))
    print(corr)
    print(pval)

    # just return the correlation
    # return corr

    # YOUR CODE HERE

    df = pd.read_csv('C:/Python/Jupyter/coursera/course1/NISPUF17.csv')
    df1 = df[['HAD_CPOX', 'P_NUMVRC']].dropna()
    df1 = df1[df1['HAD_CPOX'].lt(3)]
    print(df1.head(10))

    corr, pval = stats.pearsonr(df1['HAD_CPOX'], df1['P_NUMVRC'])
    print(corr)
    print(pval)

    return corr

    raise NotImplementedError()

assert -1<=corr_chickenpox()<=1, "You must return a float number between -1.0 and 1.0."

#


