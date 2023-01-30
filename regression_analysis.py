import numpy as np
import pandas as pd
import seaborn as sns

import plotly.offline as py
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from IPython.display import HTML

from plotly.offline import init_notebook_mode

import statsmodels.api as sm
from statsmodels.formula.api import ols

import scipy as sp

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import warnings
init_notebook_mode(connected=True)


warnings.filterwarnings('ignore')

# Problem Definition: A wine dataset is provided. The task is to analyze data and
# build a regression model to predict
# the quality of the wine.

PATH = 'data/'
filename = 'winequality-white.csv'
white_data = pd.read_csv(PATH + filename)

# Description of data
# Name of the data: Wine data from UCI Machine learning repository
# Number of data points: 4898
# Number of features: 11
# Target attribute: Quality of wine
# Range of target attribute: 3 to 9

data_head = white_data.head()
colorscale = [[0, '#4d004c'], [.5, '#f2e5ff'], [1, '#ffffff']]
df_table = ff.create_table(round(data_head.iloc[:, [0, 1, 2, 3, 4, 5]], 3), colorscale=colorscale)
py.iplot(df_table, filename='wine_quality')
df_table = ff.create_table(round(data_head.iloc[:, [6, 7, 8, 9, 10, 11]], 3), colorscale=colorscale)
py.iplot(df_table, filename='wine_quality')


# Distribution of Target Attribute
value_counts = white_data.quality.value_counts()
target_counts = pd.DataFrame({'quality': list(value_counts.index), 'value_count': value_counts})

### Frequency of target class ###
# The quality of wine ranges from 3 to 9
# The data is not balanced. The number of data points having quality 6
# is very high and quality 3 and 9 are very low.
plt.figure(figsize=(10, 4))
g = sns.barplot(x='quality', y='value_count', data=target_counts, capsize=0.3, palette='spring')
g.set_title("Frequency of target class", fontsize=15)
g.set_xlabel("Quality", fontsize=13)
g.set_ylabel("Frequency", fontsize=13)
g.set_yticks([0, 500, 1000, 1500, 2000, 2500])
for p in g.patches:
    g.annotate(np.round(p.get_height(), decimals=2),
               (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='center', xytext=(0, 10),
               textcoords='offset points', fontsize=14, color='black')

### Distribution of target variable ###
plt.figure(figsize=(10, 3))
sns.boxplot(data=white_data['quality'], orient='horizontal', palette='husl')
plt.title("Distribution of target variable")


white_data.describe().drop(columns=['quality'])

# data_head = white_data.describe().drop(columns=['quality'])
# data_head.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
#        'chlorides', 'free_SO2', 'total_SO2', 'density',
#        'pH', 'sulphates', 'alcohol']
# colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
# df_table = ff.create_table(round(data_head.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]], 3), colorscale=colorscale)
# py.iplot(df_table, filename='wine_quality')
# df_table = ff.create_table(round(data_head.iloc[:,[6,7,8,9,10]], 3), colorscale=colorscale)
# py.iplot(df_table, filename='wine_quality')

# Preprocess data
### Distribution of features ###
# If we observe the above boxplot, the range of features is different from
# each other.
# We can normalize the data. All the variables range from 0 to 1 after
# normalization and don’t lose any information.
plt.figure(figsize=(10, 10))
sns.boxplot(data=white_data.drop(columns=['quality']), orient='horizontal', palette='husl')


### Distribution of features - After normalization ###
# This looks better than before and makes it easy to understand the distribution of data.
y = white_data['quality']
white_data = white_data.loc[:, ~white_data.columns.isin(['quality'])]

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(white_data)
white_data.loc[:, :] = scaled_values

white_data['quality'] = y

data_head = white_data.head()
colorscale = [[0, '#4d004c'], [.5, '#f2e5ff'], [1, '#ffffff']]
df_table = ff.create_table(round(data_head.iloc[:, [0, 1, 2, 3, 4, 5]], 3), colorscale=colorscale, )
py.iplot(df_table, filename='wine_quality')
df_table = ff.create_table(round(data_head.iloc[:, [6, 7, 8, 9, 10, 11]], 3), colorscale=colorscale, )
py.iplot(df_table, filename='wine_quality')

columns = list(white_data.columns)
new_column_names = []
for col in columns:
    new_column_names.append(col.replace(' ', '_'))
white_data.columns = new_column_names

plt.figure(figsize=(10, 10))
sns.boxplot(data=white_data.drop(columns=['quality']), orient='horizontal', palette='husl')


# Visualize data
### Correlation between features ###

# The correlation between “density” and “residual sugar” is 0.84.
# The correlation between “alcohol” and “density” is 0.78.
# The correlation between “total sulfur dioxide” and “free sulfur dioxide” is 0.62.
# These are the three pairs of features having a high correlation(>0.5).
corr_matrix = white_data.corr().abs()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


### Distribution of each feature ###

# If we observe the distribution of all features, they follow a Normal distribution.
# There is some fluctuation in the features “sulphates” and “alcohol”.
features = white_data.copy(deep=True)
features['quality'] = y.astype('str').map(
    {'3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'})
f, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
sns.distplot(features["fixed_acidity"], rug=False, color="skyblue", ax=axes[0, 0])
sns.distplot(features["volatile_acidity"], rug=False, color="olive", ax=axes[0, 1])
sns.distplot(features["citric_acid"], rug=False, color="gold", ax=axes[0, 2])
sns.distplot(features["residual_sugar"], rug=False, color="teal", ax=axes[1, 0])
sns.distplot(features["chlorides"], rug=False, ax=axes[1, 1])
sns.distplot(features["free_sulfur_dioxide"], rug=False, color="red", ax=axes[1, 2])
sns.distplot(features["total_sulfur_dioxide"], rug=False, color="skyblue", ax=axes[2, 0])
sns.distplot(features["density"], rug=False, color="olive", ax=axes[2, 1])
sns.distplot(features["pH"], rug=False, color="gold", ax=axes[2, 2])
sns.distplot(features["sulphates"], rug=False, color="teal", ax=axes[3, 0])
sns.distplot(features["alcohol"], rug=False, ax=axes[3, 1])


upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]


### Pair plot between features ###
# This is to understand the relation between features.
# From this plot, we can see how different features are correlated with each other.
# In this plot, the features that are plotted on the x-axis and y-axis are in the given
# order itself.
features = white_data.copy(deep=True)
features['quality'] = y.astype('str').map(
    {'3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'})
sns.pairplot(features, diag_kind='kde', palette='husl', hue='quality')


### Pair plot between correlated features ###
# As we have seen above in the correlation plot, there is a high correlation(>0.5) in between some of the features.
# Here, we can visualize how these features are correlated.
# If we observe carefully, we cannot separate the data points of different quality easily,
# because all the data points of various quality are overlapped.

features = white_data.copy(deep=True)
features['quality'] = y.astype('str').map(
    {'3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'})
sns.pairplot(features, vars=to_drop, diag_kind='kde', palette='husl', hue='quality')


# Linear Regression using Gradient Descent
### Contribution of features towards target variable ###
# The method of Linear Regression that finds the coefficients of different features using
# Gradient Descent optimization, is fit to the data to see how independent variables are
# contributing to the dependent variable.
# If we observe, the coefficient of density is 7.8(absolute value), citric
# acid is 0.04, chlorides is 0.08(absolute value) and total sulfur dioxide is
# 0.12(absolute value).
# This is to understand the contribution of different features.

model_reg = LinearRegression().fit(white_data.drop(columns=['quality']), y)
y_true = white_data.quality
y_pred = model_reg.predict(white_data.drop(columns=['quality']))

column_names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                'pH', 'sulphates', 'alcohol']
regression_coefficient = pd.DataFrame({'Feature': column_names, 'Coefficient': model_reg.coef_},
                                      columns=['Feature', 'Coefficient'])

column_names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                'pH', 'sulphates', 'alcohol']

plt.figure(figsize=(15, 5))
g = sns.barplot(x='Feature', y='Coefficient', data=regression_coefficient, capsize=0.3, palette='spring')
g.set_title("Contribution of features towards target variable", fontsize=15)
g.set_xlabel("Feature", fontsize=13)
g.set_ylabel("Degree of Coefficient", fontsize=13)
g.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
g.set_xticklabels(column_names)
for p in g.patches:
    g.annotate(np.round(p.get_height(), decimals=2),
               (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='center', xytext=(0, 10),
               textcoords='offset points', fontsize=14, color='black')


    # Ordinary Least Squares(OLS)
    # In statistics, ordinary least squares (OLS) is a type of linear least squares method
    # for estimating the unknown parameters in a linear regression model.
    # The OLS method corresponds to minimizing the sum of squared differences between the
    # observed and predicted values. This minimization leads to the estimators of the
    # parameters of the model.

    # The R-squared is 0.282 and Adjusted R-squared is 0.280.
    # If p-value > 0.05, we fail to reject the null hypothesis, otherwise we reject the null
    # hypothesis.
    # The p-values of the features “citric acid” and “chlorides”, is greater than 0.05.
    # Also, the contribution of these features is very little.
    model_ols = ols("""quality ~ fixed_acidity
                        + volatile_acidity
                        + citric_acid
                        + residual_sugar
                        + chlorides
                        + free_sulfur_dioxide
                        + total_sulfur_dioxide
                        + density
                        + pH
                        + sulphates
                        + alcohol""", data=white_data).fit()

    model_summary = model_ols.summary()
HTML(
    (model_ols.summary()
     .as_html()
     .replace('<th>Dep. Variable:</th>', '<th style="background-color:#c7e9c0;"> Dep. Variable: </th>')
     .replace('<th>Model:</th>', '<th style="background-color:#c7e9c0;"> Model: </th>')
     .replace('<th>Method:</th>', '<th style="background-color:#c7e9c0;"> Method: </th>')
     .replace('<th>No. Observations:</th>', '<th style="background-color:#c7e9c0;"> No. Observations: </th>')
     .replace('<th>  R-squared:         </th>', '<th style="background-color:#aec7e8;"> R-squared: </th>')
     .replace('<th>  Adj. R-squared:    </th>', '<th style="background-color:#aec7e8;"> Adj. R-squared: </th>')
     .replace('<th>coef</th>', '<th style="background-color:#ffbb78;">coef</th>')
     .replace('<th>std err</th>', '<th style="background-color:#c7e9c0;">std err</th>')
     .replace('<th>P>|t|</th>', '<th style="background-color:#bcbddc;">P>|t|</th>')
     .replace('<th>[0.025</th>    <th>0.975]</th>',
              '<th style="background-color:#ff9896;">[0.025</th>    <th style="background-color:#ff9896;">0.975]</th>'))
)

print(model_ols.summary())

model_ols = ols("""quality ~ fixed_acidity
                        + volatile_acidity
                        + residual_sugar
                        + free_sulfur_dioxide
                        + total_sulfur_dioxide
                        + density
                        + pH
                        + sulphates
                        + alcohol""", data=white_data).fit()

model_summary = model_ols.summary()
HTML(
    (model_ols.summary()
     .as_html()
     .replace('<th>Dep. Variable:</th>', '<th style="background-color:#c7e9c0;"> Dep. Variable: </th>')
     .replace('<th>Model:</th>', '<th style="background-color:#c7e9c0;"> Model: </th>')
     .replace('<th>Method:</th>', '<th style="background-color:#c7e9c0;"> Method: </th>')
     .replace('<th>No. Observations:</th>', '<th style="background-color:#c7e9c0;"> No. Observations: </th>')
     .replace('<th>  R-squared:         </th>', '<th style="background-color:#aec7e8;"> R-squared: </th>')
     .replace('<th>  Adj. R-squared:    </th>', '<th style="background-color:#aec7e8;"> Adj. R-squared: </th>')
     .replace('<th>coef</th>', '<th style="background-color:#ffbb78;">coef</th>')
     .replace('<th>std err</th>', '<th style="background-color:#c7e9c0;">std err</th>')
     .replace('<th>P>|t|</th>', '<th style="background-color:#bcbddc;">P>|t|</th>')
     .replace('<th>[0.025</th>    <th>0.975]</th>',
              '<th style="background-color:#ff9896;">[0.025</th>    <th style="background-color:#ff9896;">0.975]</th>'))
)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def goodness(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    return mape, mse, r_squared


# model = LinearRegression().fit(white_data.drop(columns=['quality', 'citric_acid', 'chlorides']), y)
# y_true = white_data.quality
# y_pred = model.predict(white_data.drop(columns=['quality', 'citric_acid', 'chlorides']))
#
# column_names = ['fixed_acidity', 'volatile_acidity', 'residual_sugar',
#                 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
#                 'pH', 'sulphates', 'alcohol']
# regression_coefficient = pd.DataFrame({'Feature': column_names, 'Coefficient': model.coef_},
#                                       columns=['Feature', 'Coefficient'])
#
# column_names = ['fixed_acidity', 'volatile_acidity', 'residual_sugar',
#                 'free_SO2', 'total_SO2', 'density',
#                 'pH', 'sulphates', 'alcohol']
#
# plt.figure(figsize=(15, 5))
# g = sns.barplot(x='Feature', y='Coefficient', data=regression_coefficient, capsize=0.3, palette='spring')
# g.set_title("Contribution of features towards target variable", fontsize=15)
# g.set_xlabel("Feature", fontsize=13)
# g.set_ylabel("Degree of Coefficient", fontsize=13)
# g.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
# g.set_xticklabels(column_names)
# for p in g.patches:
#     g.annotate(np.round(p.get_height(), decimals=2),
#                (p.get_x() + p.get_width() / 2., p.get_height()),
#                ha='center', va='center', xytext=(0, 10),
#                textcoords='offset points', fontsize=14, color='black')


### Linearity ###
# If we observe carefully, all the partial residual plots between the independent
# and dependent variable are linear.
# Linearity condition is satisfied.
error = y_true - y_pred
error_info = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'error': error}, columns=['y_true', 'y_pred', 'error'])

fig = plt.figure(figsize=(10, 12))
fig = sm.graphics.plot_partregress_grid(model_ols, fig=fig)


### Homoskedasticity ###
# To check homoskedasticity, we plot the residuals vs predicted values/fitted values.
# If we see any kind of funnel shape, we can say that there is heteroskedasticity.

# The points are not random. Also, we can see the shape of a funnel to the right,
# which confirms that there is heteroskedasticity.
# It means that the variance of Y across all X is not the same.
# We can conclude that, Homoskedasticity condition doesn’t hold in this case.

# plt.figure(figsize=(8, 5))
# g = sns.regplot(x="y_pred", y="error", data=error_info, color='blue')
# g.set_title('Check Homoskedasticity', fontsize=15)
# g.set_xlabel("predicted values", fontsize=13)
# g.set_ylabel("Residual", fontsize=13)



### Correlation of errors ###
# If there is no correlation between errors, then the model is good.
# If we observe, there is no correlation/pattern between errors. It is purely random.

fig, ax = plt.subplots(figsize=(8, 5))
ax = error_info.error.plot()
ax.set_title('Uncorrelated errors', fontsize=15)
ax.set_xlabel("Data", fontsize=13)
ax.set_ylabel("Residual", fontsize=13)


# Normality of error terms
# This can be checked by plotting probability probability plot(p-p plot)
# or Quantile-Quantile plot(Q-Q plot).

### Probability-Probability plot ###
# If we observe the plot, we can conclude that the errors are following a
# Normal distribution, because the plot shows the fluctuation around the line and
# there is not much deviation.
# The graph is linear.
fig, ax = plt.subplots(figsize=(6, 4))
_ = sp.stats.probplot(error_info.error, plot=ax, fit=True)
ax.set_title('Probability plot', fontsize=15)
ax.set_xlabel("Theoritical Qunatiles", fontsize=13)
ax.set_ylabel("Ordered Values", fontsize=13)


### Quantile-Quantile plot ###
# ax = sm.qqplot(error_info.error, line='45')

plt.show()