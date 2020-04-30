# basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# classification / preprocessing imports
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.stats import ttest_ind
import eli5
import seaborn as sns; sns.set(style='ticks', color_codes=True)

# sl_no - Serial Number
# ssc_p - Secondary Education Percentage - 10th Grade
# ssc_b - Board of Education - Central/Others
# hsc_p - Higher Secondary Education Percentage - 12th Grade
# hsc_b - Board of Education - Central/Others
# hsc_s - Specialization in Higher Secondary Education
# degree_p - Degree Percentage
# degree_t - Degree Type
# workex - Work Experience
# etest_p - Employability Test Percentage
# specialistation - Post Graduation
# mba_p - MBA Percentage
# status - Status of Placement
# salary - salary offered by corporate to candidates

dataset = pd.read_csv('data/Placement_Data_Full_Class.csv')

# check if there are any missing values
dataset.isnull().any()
dataset['salary'].head()
# fill nan with 0's
dataset.fillna(value = 0, axis = 1, inplace = True)

""" DATA EXPLORATION """

# show the histogram to get to know the data
dataset.hist(bins = 50, figsize = (20, 15))
plt.show()

###### which degree title/specialization is much demanded by corporate? #######
""" Analysis """

placement_data = pd.DataFrame(index = dataset[dataset['salary'] > 0]['degree_t'].value_counts().index)
placement_data['Amount'] = dataset[dataset['salary'] > 0]['degree_t'].value_counts().values
placement_data.plot.pie(y = 'Amount', figsize = (5, 5))
# as in the chart, we can clearly see that majority of the corporates hire 
# people who graduated in Commercial & Management

dataset_working = dataset[dataset['status'] == 'Placed']
# return the amount of people working from each degree_t category
working_scitech = len(dataset_working[dataset_working['degree_t'] == 'Sci&Tech']['degree_t'])
working_commmgmt = len(dataset_working[dataset_working['degree_t'] == 'Comm&Mgmt']['degree_t'])
working_others = len(dataset_working[dataset_working['degree_t'] == 'Others']['degree_t'])

degree_distrib = dataset['degree_t'].value_counts()
# calculate what percentage of people of each degree_t category got a job
working_scitech_placed_per = working_scitech / degree_distrib['Sci&Tech'] * 100
working_commmgmt_placed_per = working_commmgmt / degree_distrib['Comm&Mgmt'] * 100
working_others_placed_per = working_others / degree_distrib['Others'] * 100
print(working_scitech_placed_per)
print(working_commmgmt_placed_per)
print(working_others_placed_per)
# People from Sci-Tech and Comm-Mgmt are both at 70% employability


print(dataset.groupby('degree_t')['degree_p'].mean())
# it doesn't seem like people studying Comm&Mgmt are significantly 
# more intelligent than the ones studying Sci&Tech on average

# does the specialisation determine who get the job?
specialisation_distrib = dataset['specialisation'].value_counts()
working_mkthr = len(dataset_working[dataset_working['specialisation'] == 'Mkt&HR'])
working_mktfin = len(dataset_working[dataset_working['specialisation'] == 'Mkt&Fin'])

working_mkthr_per = working_mkthr / specialisation_distrib['Mkt&HR'] * 100
working_mktfin_per = working_mktfin / specialisation_distrib['Mkt&Fin'] * 100
print(working_mkthr_per)
print(working_mktfin_per)
# Mkt&HR: nearly 56% of people got a job
# Mkt&Fin: 79% people got a job 

dataset_not_working = dataset[dataset['status'] == 'Not Placed']
not_working_mkthr = len(dataset_not_working[dataset_not_working['specialisation'] == 'Mkt&HR'])
not_working_mktfin = len(dataset_not_working[dataset_not_working['specialisation'] == 'Mkt&Fin'])

labels = ['Mkt&HR', 'Mkt&Fin']
working = [working_mkthr, working_mktfin]
not_working = [not_working_mkthr, not_working_mktfin]
width = 0.35

""" Visualisation of specialisation significance """

fig, ax = plt.subplots()

ax.bar(labels, working, width, label='Working')
ax.bar(labels, not_working, width, bottom=working,
       label='Not working')

ax.set_ylabel('Amount of people')
ax.set_xlabel('Specialisation')
ax.set_title('Working to not working ratio')
ax.legend()

plt.show()

# It seems like the specialisation is a quite significant factor to getting a job

# do your secondary/degree/higher secondary/employability test/mba test scores matter? #

""" Visualisation of work placement by score for each score """
percentage_data = dataset[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'status', 'workex']]
columns = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
names = ['Secondary Education', 'Higher Secondary Education', 'Degree', 'Employability Test', 'MBA']
colours = ['#89e5eb', '#6f93e7', '#9a26d9', '#d926cd', '#c92c5e', '#d12e2e']

for i, column in enumerate(columns):
    plt.figure(figsize = (12, 8))
    plt.scatter(x = percentage_data[column], y = percentage_data['status'], c = colours[i])
    plt.title(f'{names[i]} test score influence on status', fontsize = 15)
    plt.xlabel(f'{names[i]} % score', fontsize = 15)
    plt.ylabel('Status', fontsize = 15)
    plt.show()

# from the plots, it looks like the most impact have grades from
# Secondary/Higher Secondary and Degree tests, at least from them it's easier to 
# determine who is going to be hired

# let's take a mean of the most concrete scores
percentage_data['mean_of_score'] = percentage_data[['ssc_p', 'hsc_p', 'degree_p']].mean(axis = 1)

""" Visualisation of work placement by mean of scores """

plt.figure(figsize = (12, 8))
plt.scatter(x = percentage_data['mean_of_score'], y = percentage_data['status'])
plt.title('Mean score influence on status', fontsize = 15)
plt.xlabel('Mean score of first 3 scores', fontsize = 15)
plt.ylabel('Status', fontsize = 15)
plt.show()
# this one is the most descriptive so far, in my opinion

############# does your previous work experience matter? ######################

""" Analysis """

people_placed = dataset[dataset['salary'] > 0]
people_placed['workex'].value_counts()
# seems like corporates are indifferent to the previous work experience

workex_job_p = len(dataset[(dataset['workex'] == 'Yes') & (dataset['salary'] > 0)]) / len(dataset[dataset['workex'] == 'Yes'])
no_workex_job_p = len(dataset[(dataset['workex'] == 'No') & (dataset['salary'] > 0)]) / len(dataset[dataset['workex'] == 'No'])
# however, 86.5 % of people who had previous work experience got a job and 
# nearly 60% of those who hadn't had previous work experience got a job as well

""" PREDICTING """

num_data = []
cat_data = []

for i, column in enumerate(dataset.dtypes):
    if column == object:
        cat_data.append(dataset.iloc[:, i])
    else:
        num_data.append(dataset.iloc[:, i])

num_data = pd.DataFrame(num_data).transpose()
cat_data = pd.DataFrame(cat_data).transpose()

y = cat_data['status']
cat_data = cat_data.drop(['status'], axis=1)
# num_data is in a right shape and form, no need to transform it

# one hot encoding
cat_data_encoded = pd.get_dummies(cat_data)

final_data = pd.concat([num_data, cat_data_encoded], axis = 1)

X = final_data.drop(['salary', 'sl_no'], axis=1)

""" TASKS FROM KAGGLE """

""" Association between 'mba_p' (outcome) and 'degree_p' (input) """
X_1 = dataset['degree_p']
y_1 = dataset['mba_p']
X1 = np.array(X_1).reshape(-1, 1)
y1 = np.array(y_1).reshape(-1, 1)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.25)

lin_reg = LinearRegression(n_jobs=-1)
lin_reg.fit(X1_train, y1_train)
y_pred = lin_reg.predict(X1_test)

mse = mean_squared_error(y1_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Visualising train set
plt.scatter(x=X1_train, y=y1_train, c='green')
plt.plot(X1_train, lin_reg.predict(X1_train))
plt.xlabel('degree_p')
plt.ylabel('mba_p')
plt.title('Train set')
plt.show()

# Visualising test set
plt.scatter(x=X1_test, y=y1_test, c='red')
plt.plot(X1_test, lin_reg.predict(X1_test))
plt.xlabel('degree_p')
plt.ylabel('mba_p')
plt.title('Test set')
plt.show()

print(f"Coefficient: {lin_reg.coef_[0][0]}")
# pretty low coefficient, low linear dependency

""" Multiple Lin Reg with 'mba_p'as response var, 'ssc_p' and 'hsc_p' as predictor var """

# get data
data_task = dataset[['ssc_p', 'hsc_p', 'mba_p']]
X2 = np.array(data_task.iloc[:,:2])
y2 = np.array(data_task.iloc[:, 2])

# train test split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25)
lin_reg = LinearRegression(n_jobs=-1)

# fit features and labels
lin_reg.fit(X2_train, y2_train)
y2_pred = lin_reg.predict(X2_test)

lin_reg.coef_
lin_reg.intercept_
# linear regression equation
# mba_p = 45.3 + 0.168*ssc_p + 0.08*hsc_p

# backward elimination to check significance
X2 = np.append(arr = np.ones(shape=(215, 1)).astype(float), values=X2, axis=1)
X_opt = X2[:, [0, 1, 2]]
regressor_OLS = sm.OLS(endog=y2, exog=X_opt).fit()
regressor_OLS.summary()
print('SSC_P pvalue: ', regressor_OLS._results.pvalues[1])
print('HSC_P pvalue: ', regressor_OLS._results.pvalues[2])

""" Histogram of degree_p """

# 1. 
dataset['degree_p'].hist()

# 2. Differentiate based on status
dataset[dataset['status'] == 'Placed']['degree_p'].hist(color='green', bins=15, alpha=0.5)
dataset[dataset['status'] == 'Not Placed']['degree_p'].hist(color='red', bins=15, alpha=0.5)
plt.show()

# 3. Add gender difference
grid = sns.FacetGrid(dataset, col='status', row='gender')
grid = grid.map(plt.hist, 'degree_p', color='r')

# 4. Boxplot for degree_p
