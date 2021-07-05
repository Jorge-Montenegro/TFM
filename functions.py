#-----------------------------------------ALL LIBRARIES NEEDED-------------------------------------#

#Array and Frames Libraries
import pandas as pd
import numpy as np
#Plotting Libraries
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import seaborn as sns
#Analyze distributions
from fitter import Fitter
#Machine Learning Libraries
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
#Model Prophet
from prophet import Prophet
#Model Keras
from keras import models, layers, metrics, losses, optimizers
#Model Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
#Normalization
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler, MinMaxScaler
#Scoring
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
#Selection
from sklearn.feature_selection import VarianceThreshold
#Statistical Models
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from patsy import dmatrices
#Regular Expressions Libraries
import re
#Avoid warnings
import warnings
warnings.filterwarnings('ignore')
#Progress Bar
from tqdm.notebook import tqdm
#Serilizing
import pickle
#Combination
import itertools as it

#--------------------------------------------------------------------------------------------------#

#--------------------------------FUNCTIONS FOR VISUALIZATION PURPOSE-------------------------------#    

#This is a simple method for drawing a plot x=numbers of observations and y=target values
def draw_plot(df, column):
    """
    DESCRIPTION
      This draw a plot and add a title
    ARGUMENTS
      df: This is the DataFrame where the data is stored
      column: The date column name
    RETURN
      One plot
    """
    
    #Prepare the Axes and size
    fig, ax = plt.subplots(1, 1)
    
    #Draw the plot with the right titles and labels
    ax.plot(df.index, df[column])
    ax.set_title(f'{column} Evolution')
    ax.set_xlabel('Obervations Range') 
    ax.set_ylabel(column)

#--------------------------------------------------------------------------------------------------#

#This show a scatter plot to see the mean and std relationship - Additive or Multiplicative
def draw_decomposition(df):
    """
    DESCRIPTION
      This draw a scatter plot x-axis: std and y-axis: mean and adjust with one line
      if the slope is around or more than 45º means the decomposition is multiplicative otherwise additive
    ARGUMENTS
      df: This is the DataFrame where the data is stored
    RETURN
      One plot
    """
    
    #Calculate the mean and std
    #data_small.groupby(['Year'])['Revenue'].std()
    mean_year = df.groupby(df.columns[0:1][0])[df.columns[-1]].mean()
    std_year = df.groupby(df.columns[0:1][0])[df.columns[-1]].std()
    
    #Create a new DataFrame with the Mean and Std per year
    data = pd.DataFrame([mean_year, std_year], index= ['Mean', 'Std']).T
    data.drop(data.index[6], inplace= True) # Remove 2021
    
    #Prepare the Axes and size
    sns.regplot(data['Mean'], data['Std']).set(title= 'Decomposition Analysis')
    
#--------------------------------------------------------------------------------------------------#    

#Quick chart for checking Autocorrelation and Partial Autocorrelation
def draw_autocorrelation(df, lags):
    """
    DESCRIPTION
      This draw an autocorrelation or partial autocorrelation chart based on parameter
    ARGUMENTS
      df: DataFrame where the data is stored
      lags: Number of n-1 observations
    RETURN
      Two plots
    """
    
    #Two columns, one row
    fig, ax = plt.subplots(1, 2)
    #Let's convert NxM Array into N
    ax = ax.flat
    
    plot_acf(df[df.columns[-1]], lags= lags, ax= ax[0])
    plot_pacf(df[df.columns[-1]], lags= lags, ax= ax[1])
    
    for i in range(len(ax)):
        #Horizontal lines around 0.2 correlation threshold
        ax[i].hlines(0.2, 0, lags, color= 'r', linestyles= 'dotted', linewidths= 3, alpha=0.5)
        ax[i].hlines(-0.2, 0, lags, color= 'r', linestyles= 'dotted', linewidths= 3, alpha=0.5)
    
    plt.show()

#--------------------------------------------------------------------------------------------------#

#This method draw for the target variable three different histograms versions, regular, logarithmic and square
def draw_set_of_distributions(df, column):
    """
    DESCRIPTION
      This function draw three different axes with these type of distributions:
      1.- Regular Distribution
      2.- Logarithmic Distribution
      3.- Square Distribution
    ARGUMENTS
      df: This is the DataFrame where the data is stored
      column: The date column name
    RETURN
      Three different frames
    """
    
    #Dictionary of distributions
    #It is a Tuple with the Array and Axes
    distributions = dict({
        'Regular Distribution': (df[column], 0),
        'Logarithmic Transformation': (np.log1p(df[column]), 1),
        'Square Transformation': (np.sqrt(df[column]), 2)
    })
    
    #Prepare the Axes and size
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(20,20)
    
    #Get the bins for the variable
    bins = get_bins(df, column)
    
    #Draw the different distributions
    for distribution in distributions:
        #Get the values and axe for each Distribution
        values = distributions[distribution][0]
        axe = distributions[distribution][1]
        #Draw the Distribution and Title
        sns.histplot(values, bins= bins, kde= True, stat= "density", ax= ax[axe])
        sns.rugplot(values, ax= ax[axe])
        #sns.distplot(values, hist= True, bins= bins, rug= True, ax= ax[axe])
        ax[axe].set_title(distribution)

#--------------------------------------------------------------------------------------------------#

#This method draw a grid with a Histogram by feature
def draw_distribution(df, columns):
    """
    DESCRIPTION
      This function draw a distribution analysis per feature
    ARGUMENTS
      df: This is the DataFrame where the data is stored
      column: This is the column list to draw
    RETURN
      Multiple distribution based on the columns values
    """
    
    #We need to transform the parameter into a list in case there is only one column
    columns = list(columns)
    
    #Prepare the Axes and size
    #Always, it will be 3 columns by rows based on the numbers of variables
    rows = 1 + int(len(columns)/3) if len(columns) % 3 == 2 else len(columns) % 3 + int(len(columns)/3)
    fig, ax = plt.subplots(rows, 3)
    #Let's convert NxM Array into N
    ax = ax.flat
    fig.set_size_inches(20,20)
    
    #Iterate in all columns
    for i, column in enumerate(columns):
        #Get the bins for the variable
        bins = get_bins(df, column)
        #if no bins, let's use by default
        if bins == 0:
            sns.histplot(df[column], kde=True, stat= "density", ax= ax[i])
        else:
            sns.histplot(df[column], bins= bins, kde=True, stat= "density", ax= ax[i])
        ax[i].set_title(column)
    fig.tight_layout()
    plt.subplots_adjust(top= 0.92)
    fig.suptitle('Features Distributions', fontsize= 25)

#--------------------------------------------------------------------------------------------------#

def draw_distribution_evolution(df, column, condition):
    """
    DESCRIPTION
      This function draw how the target variable has envolved across the time
    ARGUMENTS
      df: This is the DataFrame where the data is stored
      column: This is the column list to draw
    RETURN
      Multiple distribution based on the columns values
    """
    
    #We need to transform the parameter into a list in case there is only one column
    pivots = list(df[condition].unique())
    
    #Prepare the Axes and size
    #Always, it will be X columns by rows based on the numbers of variables. If we put x=2 means n row by 2 columns, 
    #x=3 n rows by 3 columns and so on
    x = 2
    rows = 1 + int(len(pivots)/x) if len(pivots) % x == (x-1) else len(pivots) % x + int(len(pivots)/x)
    fig, ax = plt.subplots(rows, x)
    #Let's convert NxM Array into N
    ax = ax.flat
    fig.set_size_inches(20,30)
    
    #Colors available - 31 possible different options in case we want to see the distribution by Day
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 
             'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b']
    
    #Iterate in all pivots
    for i, pivot in enumerate(pivots):
        for j in range(i+1):
            #Get the bins for the variable
            bins = np.histogram_bin_edges(df[df[condition] == pivots[j]][column], bins= 'stone')
            sns.histplot(df[df[condition] == pivots[j]][column], bins= bins, 
                         kde= True, stat= "count", ax= ax[i], color= colors[j], alpha= 0.2)
            
            #Vertical line with the median
            #Get the most common value
            ymax = (np.histogram(df[df[condition] == pivots[j]][column], bins= bins, density= False)[0].max()) * 1.25
            ax[i].vlines(df[df[condition] == pivots[j]][column].median(), 0, ymax, color= colors[j], 
                   linestyles= 'solid', linewidths= 2, alpha=0.9)
        
        #Title
        title = 'From ' + str(pivot)
        if len(range(i)) > 0:
            title = 'From ' + str(pivots[0]) + ' to ' + str(pivot)
        ax[i].set_title(title)
            
    fig.tight_layout()
    plt.subplots_adjust(top= 0.95)
    fig.suptitle(f'{column} distribution across {condition}', fontsize= 25)

#--------------------------------------------------------------------------------------------------#

#This draw a correlation grid Target vs. Feature
def draw_correlation(df, columns, target):
    """
    DESCRIPTION
      This function draw the correlation between each feature and the objective
    ARGUMENTS
      df: This is the DataFrame where the data is stored
      column: This is the column list with all features
      target: This is the target value to correlate which each feature
    RETURN
      Multiple regression linear plots for pair - target vs feature
    """
    
    #We need to transform the parameter into a list in case there is only one column
    columns = list(columns)
    
    #Prepare the Axes and size
    #Always, it will be 3 columns by rows based on the numbers of variables
    rows = 1 + int(len(columns)/3) if len(columns) % 3 == 2 else len(columns) % 3 + int(len(columns)/3)
    fig, ax = plt.subplots(rows, 3)
    #Let's convert NxM Array into N
    ax = ax.flat
    fig.set_size_inches(20,20)
    
    #Iterate in all columns
    for i, column in enumerate(columns):
        sns.regplot(x= df[column], y= df[target], marker= 'x', 
                scatter_kws= {"alpha": 0.4}, line_kws= {"color": "r","alpha": 0.7}, ax= ax[i])
        ax[i].set_title(f'{column} correlation with {list(target)[0]}')
        
    fig.tight_layout()
    plt.subplots_adjust(top= 0.92)
    fig.suptitle('Correlation Analysis', fontsize= 25)

#--------------------------------------------------------------------------------------------------#

#This function gives you the right bins proportion
def get_bins(df, column):
    """
    DESCRIPTION
      This function returns the right proportion of bins needed
    ARGUMENTS
      df: This is the DataFrame to calculate the bins
      column: The date column name
    RETURN
      The h of the bins
    """
    
    #Formula is Max Value - Min Value / Number of Observations
    return int((df[column].max() - df[column].min()) / len(df[column]))

#--------------------------------------------------------------------------------------------------#

#This function is not needed-----------------------------------------------------------------------#
def max_closest_number(n, x= 1):
    if int(n // x) <= 10:
        return (int(n // x) + 1) * x
    else:
        return max_closest_number (n, x * 10)

#--------------------------------------------------------------------------------------------------#

#This function return a correlation Heatmap
def corr_matrix (df):
    """
    DESCRIPTION
      This function draw the correlation matrix across the variables
    ARGUMENTS
      df: This is the DataFrame where the data is stored
    RETURN
      A Heatmap
    """
    
    #Calculate correlation
    corr = df.corr()
    #Keep just diagonal and below
    data = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
    #Heatmap Size - plt.subplots(figsize=(20, 10)) - fig.set_size_inches(20,10)
    f, ax = plt.subplots(figsize=(20, 10))
    #Background color, it has to set at axe level because it is not working if we change axes.facecolor value
    #and then set a style because override that value
    ax.set_facecolor('white')
    ax = sns.heatmap(data, cmap='Spectral', annot= True, center= -0.1)

#--------------------------------------------------------------------------------------------------#

#This function shows the differences between real y and predicted ŷ
def draw_predict(model, X_real, y_real):
    """
    DESCRIPTION
      This plot real values versus predicted values
    ARGUMENTS
      model: Regression model to apply predict
      X_real: Real features
      y_real: Real objective or dependant variable
    RETURN
      Draw a plot
    """
    
    #Get the predictions
    y_pred = model.predict(X_real)
    print(f'{type(model).__name__} | Predict MAE: {mean_absolute_error(y_real,y_pred):.4f}')
    print(f'{type(model).__name__} | Predict RMSE: {np.sqrt(mean_squared_error(y_real,y_pred)):.4f}')
    print(f'{type(model).__name__} | Predict R2: {r2_score(y_real,y_pred):.4f}')
    
    #Plot the real Ys versus Y predicted
    fig = plt.figure()
    fig.set_size_inches(20,10)
    plt.plot(range(X_real.iloc[:,0].count()), y_real)
    plt.plot(range(X_real.iloc[:,0].count()), y_pred, c='r', linewidth=3)

#--------------------------------------------------------------------------------------------------#

#Draw the target variable and adjusted with a polynomial
def draw_least_squares_polynomial(df, column, degree):
    """
    DESCRIPTION
      This function draws a polynomial based on DataFrame, target column and polynomial degree
    ARGUMENTS
      df: This is the DataFrame where the data is stored
      column: This is the value to adjust
      degree: degree of the fitting polynomial
    RETURN
      Draw a plot
    """
    
    #Calculate the parameters
    x = df.reset_index().index
    #Lets prepare a polynomial regression line for Revenue
    y = df[column]
    polynomial_coef = np.polyfit(x, y, degree)
    
    #Prepare the Axes and size
    fig, ax = plt.subplots(1, 1)
    
    #Draw the plot with the right titles and labels
    ax.set_title(f'{column} and Polynomial Adjustment')
    ax.set_xlabel('Obervations Range') 
    ax.set_ylabel(column)
    plt.plot(df.index, y)
    plt.plot(df.index, backtracking_polynomial_parameters(polynomial_coef, len(polynomial_coef), x), c= 'r', linewidth= 3)

#--------------------------------------------------------------------------------------------------#

#Get polynomial values
def backtracking_polynomial_parameters(l, i, x):
    """
    DESCRIPTION
      This function return a vector of coefficients for plotting easily. It is using backtracking
    ARGUMENTS
      l: list of coefficients from np.polyfit, it does not matter the degree
      i: This is the len of the list
      x: It is the X parameter
    RETURN
      A value p[0] * x**deg + ... + p[deg]
    """
    if i == 0:
        return 0
    else:
        #Empiezo de atrás hacia adelante porque el primer polinomio es x^y...hasta b
        #Si el índice tiene 3 elementos entonces x²+x+b y así sucesivamente
        return l[-i]*x**(i-1) + backtracking_polynomial_parameters(l, i-1, x)

#--------------------------------------------------------------------------------------------------#

#Draw a grid with Residual Diagnosis
def draw_residuals_diagnosis(df, columns):
    """
    DESCRIPTION
      This function draws a grid with four elements about residuals
    ARGUMENTS
      df: DataFrame where data is located
      columns: This has target, predict and residuals
    RETURN
      Draw a grid with all residuals analysis
    """
    
    #Just a pointer of the DataFrame
    data = df
    
    #Get columns name
    real = columns[0]
    predicted = columns[1]
    residuals = columns[2]
    
    fig, ax = plt.subplots(2, 2)
    #Let's convert NxM Array into N
    ax = ax.flat
    
    #First axe
    ax[0].scatter(data[real], data[predicted])
    ax[0].plot([data[real].min(), data[real].max()], [data[real].min(), data[real].max()], 
             color= 'r', ls= 'dotted', lw= 2, alpha=0.5)
    ax[0].set_xlabel(real)
    ax[0].set_ylabel(predicted)
    ax[0].set_title(f'{predicted} vs {real}')
    
    #Second axe
    ax[1].scatter(data.index, data[residuals])
    ax[1].hlines(0, data.index.min(), data.index.max(), color= 'r', linestyles= 'dashed', linewidths= 3, alpha=1)
    ax[1].set_xlabel('Dates')
    ax[1].set_ylabel(residuals)
    ax[1].set_title(f'{residuals} of the model')
    
    #Third axe
    data[residuals].plot(kind='kde', ax= ax[2])
    ax[2].set_xlabel(residuals)
    ax[2].set_title(f'{residuals} Distribution')
    
    #Fourth axe
    sm.qqplot(data[residuals], fit= True, line= 'q', color= 'firebrick', alpha= 0.4, ax= ax[3]);
    
    fig.tight_layout()
    plt.subplots_adjust(top= 0.9)
    fig.suptitle('Residual Diagnosis', fontsize= 25)

#--------------------------------------------------------------------------------------------------#    

#Return the confidence ratio
def get_confidence(ratio):
    """
    DESCRIPTION
      This function returns the ratio for caulculating the upper and lower confidence intervals
    ARGUMENTS
      ratio: from 50% to 99%
    RETURN
      Return the confidence ratio
    """
    
    confidence = {
        '50%': 0.67,
        '60%': 0.84,
        '70%': 1.04,
        '80%': 1.28,
        '90%': 1.64,
        '95%': 1.96,
        '99%': 2.58
    }
    
    return confidence[ratio]

#Get the confidence intervals for drawing
def get_confidence_intervals(forecast_train, forecast_test, column, ratio):
    """
    DESCRIPTION
      This function returns the lower and upper confidence intervals considering a Normal Distribution
    ARGUMENTS
      forecast_train: Ys from the target Dataset
      forecast_test: Ys from the test Dataset
      column: This is the target name
      ratio: Confidence interval ratio
    RETURN
      Return a new DataFrame with the new two columns lower and upper
    """
    
    confidence = {
        '50%': 0.67,
        '60%': 0.84,
        '70%': 1.04,
        '80%': 1.28,
        '90%': 1.64,
        '95%': 1.96,
        '99%': 2.58
    }
    
    #Let's copy the model with the target and predict columns
    intervals = forecast_test.copy()
    #Get the 95% confidence interval - https://otexts.com/fpp2/prediction-intervals.html
    #From train model, calculate the std of the residuals
    intervals['Residuals_std'] = get_confidence(ratio) * forecast_train['Residuals'].std()
    #Calculate the lower and upper range
    intervals[f'{column}_lower'] = intervals[f'{column}_predicted'] - intervals['Residuals_std']
    intervals[f'{column}_upper'] = intervals[f'{column}_predicted'] + intervals['Residuals_std']

    return intervals    

#--------------------------------------------------------------------------------------------------#

#This draw the a horizontal bars for checking the best model
def draw_barh(df, column):
    """
    DESCRIPTION
      This function draw a horizontal bar sort by lowest to highest kpi. Normally, RMSE or R2
    ARGUMENTS
      df: DataFrame where data is located
      column: This is the target name
    RETURN
      Draw a horizontal bar sort by low to high
    """
    
    ax = df[column].sort_values(ascending = False).plot(kind='barh')
    for i, name in enumerate(list(df[column].sort_values(ascending = False).index)):
        col_value = df.loc[name, column]
        ax.annotate(f'{col_value:.2f}', (col_value, i), xytext=(10, -5), textcoords='offset points')
    
    ax.set_title(f'Model Comparison by {column}')
    
#--------------------------------------------------------------------------------------------------#

#------------------------------FUNCTIONS FOR DATA MANIPULATION PURPOSE-----------------------------#

#Clean the zero Revenue in the original DataFrame
def fill_zero_revenue(df):
    """
    DESCRIPTION
      This function complete the zero values in Revenue pero tracking issues in Omniture. We are going to use the
      mean between date after and before and apply this new Revenue to Direct Program
    ARGUMENTS
      df: This is the DataFrame to clean
    RETURN
      A new DataFrame with the Revenue zero values cleaned
    """
    
    #Clone the current DataFrame
    data = df.copy()
    
    #Let's calculate what dates have zero Revenue because we are using that values for pivoting
    temp_df = data.groupby(['Date'])['Revenue'].sum().reset_index()
    zero_revenue_dates = temp_df[temp_df['Revenue'] == 0]['Date']
    
    #Prepare the range of dates +1 and -1 Day considering the data with zero Revenue
    date_after = zero_revenue_dates - np.timedelta64(-1,'D')
    date_before = zero_revenue_dates - np.timedelta64(1,'D')
    
    #Create Series with the Revenue
    revenue_after = data[data['Date'].isin(date_after)].groupby('Date')['Revenue'].sum()
    revenue_before = data[data['Date'].isin(date_before)].groupby('Date')['Revenue'].sum()
    
    #Create a DataFrame with the two Revenue Columns -1 and +1 and calculate the mean
    revenue_df = pd.DataFrame([revenue_before.values, revenue_after.values], index= ['after', 'before']).T
    revenue_mean = revenue_df.mean(axis=1)
    
    data.loc[(data['Date'].isin(zero_revenue_dates)) & (data['Program'] == 'Direct'), 'Revenue'] = revenue_mean.values
    
    return data

#--------------------------------------------------------------------------------------------------#

#Convert Date into Year, Month and Day
def split_dates_by_columns(df, column):
    """
    DESCRIPTION
      This function returns a new DataFrame extracting year, month and day from a date column
    ARGUMENTS
      df: This is the DataFrame to manipulate
      column: The date column name
    RETURN
      A new DataFrame with 3 new columns related to the date and removing the Date column
    """
    
    #This is the returned DataFrame
    data = pd.DataFrame()
    
    #Add the new columns
    data['Year'] = df[column].dt.year
    data['Month'] = df[column].dt.month
    data['Day'] = df[column].dt.day
    
    #Let's take all columns except the Date one
    columns = set(df.columns) - set([column])
    
    #Return the merge Dataframes with the Date Index
    data = pd.concat([data, df[columns]], axis= 1)
    data.set_index(df[column], inplace= True)
    return data

#--------------------------------------------------------------------------------------------------#

#This function is not needed-----------------------------------------------------------------------#
#Convert Date into ISO Year, ISO Week and Day
def split_dates_by_iso_columns(df, column):
    """
    DESCRIPTION
      This function returns a new DataFrame extracting ISO year, ISO week and day from a date column
    ARGUMENTS
      df: This is the DataFrame to manipulate
      column: The date column name
    RETURN
      A new DataFrame with 3 new columns related to the date and removing the Date column
    """
    
    #This is the returned DataFrame
    data = pd.DataFrame()
    
    #Invoke the method which creates a new DataFrame with Year, Week and Day and Capitalize the Header
    data = df[column].dt.isocalendar()
    data.columns = data.columns.str.capitalize()
    
    #Let's take all columns except the Date one
    columns = set(df.columns) - set([column])
    
    #Return the merged Dataframes
    return pd.concat([data, df[columns]], axis= 1)

#--------------------------------------------------------------------------------------------------#

#This function is not needed-----------------------------------------------------------------------#
#Group some traffic sources into Direct and Display based on some specific rules
def group_programs(df):
    """
    DESCRIPTION
      This function returns a new DataFrame just keeping the main programs: Direct, SEO, SEM, Affiliate, Email and Display
    ARGUMENTS
      df: This is the DataFrame to group
    RETURN
      A new DataFrame with only those Programs
    """
    
    #Create a copy of the DataFrame
    data = df.copy()
    
    #Now, we have two main sets, selected_programs what we want to keep
    #And sub_programs which contain the remaining Programs to merge with two Direct and Display
    all_programs = set(data['Program'].unique())
    selected_programs = set(['Direct', 'SEO', 'SEM', 'Affiliate', 'Email', 'Display'])
    sub_programs = all_programs - selected_programs
    
    #We are going to create two subsets one which will have all Programs to merge with Direct
    #And another one to merge with Display
    new_display = set()
    for i in sub_programs:
        if 'Social' in i:
            new_display.add(i)
    new_direct = sub_programs - new_display
    
    #Just rename all new_direct Programs as Direct and same with Display
    data.loc[(data['Program'].isin(list(new_direct))), 'Program'] = 'Direct'
    data.loc[(data['Program'].isin(list(new_display))), 'Program'] = 'Display'
    
    #Return the DataFrame with the new Programs cleaned
    return data

#--------------------------------------------------------------------------------------------------#

#This function is not needed-----------------------------------------------------------------------#
#Add two new qualitative columns, Black Friday=1 and Cyber Monday=1
def get_black_friday_cyber_monday(df):
    """
    DESCRIPTION
      This function returns a new DataFrame with a new Black Friday and Cyber Monday column
    ARGUMENTS
      df: This is the DataFrame with Year and Week columns
    RETURN
      A new DataFrame with 2 new columns
    """
    
    #Create a copy of the DataFrame
    data = df.copy()
    #These are the dictionaries when a Black Friday or Cyber Monday takes place
    bf_dict = {47:[2016, 2017, 2018, 2021],
          48:[2015, 2019, 2020]}
    cm_dict = {48:[2016, 2017, 2018, 2021],
          49:[2015, 2019, 2020]}
    
    data['Black_Friday'] = 0
    data['Cyber_Monday'] = 0
    #
    for week in bf_dict:
        data['Black_Friday'][data['Year'].isin(bf_dict[week]) & (data['Week'] == week)] = 1
    for week in cm_dict:
        data['Cyber_Monday'][data['Year'].isin(cm_dict[week]) & (data['Week'] == week)] = 1
    
    return data

#--------------------------------------------------------------------------------------------------#

#Create a new DataFrame with the BlackFriday, Easter and Covid columns
def get_exogenous_features(df):
    """
    DESCRIPTION
      This function returns a new DataFrame with a new Black Friday Week, Easter and Covid
    ARGUMENTS
      df: This is the original DataFrame to transform
    RETURN
      A new DataFrame with 3 new columns
    """
    
    #Constants of the seasonal variables
    bf_date_list = ['2015-11-27', '2016-11-25', '2017-11-24', '2018-11-23', '2019-11-29', '2020-11-27', '2021-11-26']
    easter_date_list = ['2015-03-30', '2016-03-21', '2017-04-10', '2018-03-26', '2019-04-15', '2020-04-06', '2021-03-29']
    covid_date_range = ['2020-03-14', '2020-06-21']
    
    #Create a new DataFrame same Index
    data = pd.DataFrame(index= df.index)
    
    #To initialize the features
    data['Black_Friday'] = 0
    data['Easter'] = 0
    data['Covid'] = 0
    
    #Create the list of date ranges
    bf_week_dates = create_date_range(bf_date_list, 'D', 4)
    easter_week_dates = create_date_range(easter_date_list, 'D', 8)
    
    #Apply the 1's values like OneHotEncoding
    for range_date in bf_week_dates:
        data['Black_Friday'][data.index.isin(range_date)] = 1
       
    for range_date in easter_week_dates:
        data['Easter'][data.index.isin(range_date)] = 1
    
    data['Covid'].loc[covid_date_range[0]:covid_date_range[1]] = 1
    
    #Return new DataFrame
    return pd.concat([df[df.columns[0:3]], data, df[df.columns[3:]]], axis= 1)

#--------------------------------------------------------------------------------------------------#

#Create the lagged variables for a specific target
def get_lag_features(df, column, lags):
    """
    DESCRIPTION
      This function creates a new DataFrame with the lagged variables
    ARGUMENTS
      df: This is the DataFrame from get train and test data
      column: This is the target variable
      lags: list of number of lgas
    RETURN
      A new DataFrame with the lagged variables
    """
    
    #Final DataFrame to return
    data = pd.DataFrame()
    
    #Generating the lagged terms
    for i in lags[::-1]:
        data[f'{column[0:3]}-{i}'] = df[column].shift(i)
    
    return data

#--------------------------------------------------------------------------------------------------#

#Give a Dataframe with the Year, Month, Day as orinal variables and Exogenous
def prepare_seasonal_data(df):
    """
    DESCRIPTION
      This function creates a new DataFrame with Year, Monday and Day plus Exogenous features
    ARGUMENTS
      df: This is the DataFrame from the features to transform
    RETURN
      A new DataFrame with the Dummies and Exogenous features
    """

    #Create Seasonal Features - I am transforming as if they were non-numeric variables
    data_seasonal = pd.get_dummies(df[['Year', 'Month', 'Day']].astype('str'))
    #Create Exogenus Features
    data_exogenous = get_exogenous_features(df)[['Black_Friday', 'Easter', 'Covid']]
    #Create the model to Train and Test with the different Machine Learning models
    data_se = pd.concat([data_seasonal, data_exogenous], axis= 1)
    
    return data_se

#--------------------------------------------------------------------------------------------------#

#Create the lagged variables for training purposes
def get_lag_period(df, column, lags, period):
    """
    DESCRIPTION
      This function creates a list of new DataFrames with the lagged variables by period, for example
      if the period is 7 means, 7 days to predict so we need to lag the variable for day 2, day 3 and so on
    ARGUMENTS
      df: This is the DataFrame from get train and test data
      column: This is the target variable
      lags: list of number of lgas
      period: Number of days to predict
    RETURN
      A new list with all DataFrames with the lagged variables for training
    """

    #Convert lags into array for easy manipulation
    lags_a = np.array(lags)
    #List with all data_lagged
    data_model_list = list()

    #Create Seasonal and Exgenous Features
    data_se = prepare_seasonal_data(df)
    
    for i in range(period):
        
        #Create the model for Machine Learning models lagged by period
        #If we want to predict Day+2, we need to lag the already lagged variables +1
        data_lag = get_lag_features(df, column, lags_a + i)
        data_model = pd.concat([data_se, data_lag, df[column]], axis= 1)
        data_model.dropna(inplace= True)
        #add the model to a list
        data_model_list.append(data_model)

    return data_model_list

#--------------------------------------------------------------------------------------------------#

#Return a list with the Xs lagged at T+1 for forecasting purposes
def get_time_plus_1(df, column, lags):
    """
    DESCRIPTION
      This function creates a list with the lagged features for forecasting purposes. This pick the list of lagged and
      retreive the same value. This function is needed when your lags are not consecutive
    ARGUMENTS
      df: This is the DataFrame from get the target to lag
      column: This is the target variable
      lags: list of number of lgas
    RETURN
      A new list with the features T+1 for forecasting purposes
    """

    #Get next T+1 in Time Series for predicting
    lags_a = np.array(lags) * -1
    #List of new Xs for predicting next step - T+1
    features = list()
    #columns name for the DataFrame
    columns = list()
    
    for i in range(len(lags_a)):
        features.append(df[column].iloc[lags_a[i]])
            
    return features[::-1]

#--------------------------------------------------------------------------------------------------#

#This function return the lagged variable for predictions sorted by Data Index
def get_lag_features_forecast(df, column, lags, predictions):
    """
    DESCRIPTION
      This function calls several times the get_time_plus_1 for each predictions. For example, predictions = 7
      this returns a DataFrame with the next 7 days for predictions with the lagged variables needed to use in the
      machine learning model
    ARGUMENTS
      df: This is the DataFrame from get the target to lag
      column: This is the target variable
      lags: list of number of lgas
      predictions: Number of days for forecasting
    RETURN
      A new DataFrame with the lagged variables for the predictions
    """
    
    #List of the lags features by prediction
    lags_predict = list()
    #This is the list of lags, we will convert in array to apply T+1, T+2 and so on over the same group
    lags_a = np.array(lags)

    #This is a list with the lagged target for T+1, T+2, and so on
    #This is needed because our model predict one day T+1, another model for T+2 and so on
    for i in range(predictions):
        lags_predict.append(get_time_plus_1(df, column, lags_a + i))
    
    #Columns name for the DataFrame. Index with the extra period
    #Important, columns should be the same name
    columns = list()
    index = create_date_range([df.index[-1] + pd.to_timedelta(1, 'D')], 'D', predictions)

    for i in lags:
        columns.append(f'{column[0:3]}-{i}')

    #Create the DataFrame with the features needed for making the forecast    
    lag_predict_data = pd.DataFrame(lags_predict)
    
    #Columns and Index
    lag_predict_data.columns = columns[::-1]
    lag_predict_data.index = index[0]

    return lag_predict_data

#--------------------------------------------------------------------------------------------------#

def get_dataindex_forecast(df, predictions):
    """
    DESCRIPTION
      This function create a DataFrame with Year, Month and Day for forecasting purposes because the model need those values
    ARGUMENTS
      predictions: Number of days for forecasting
    RETURN
      A new DataFrame with the lagged variables for the predictions
    """
    
    #Get the last row and add <N-Predictions> more rows
    index_range = create_date_range([df.index[-1] + pd.to_timedelta(1, 'D')], 'D', predictions)
    
    #Create new rows, index
    row_list = list()
    for i in index_range[0]:
        row_list.append(pd.Series(name= i))
    
    date_forecast = pd.DataFrame(row_list)
    
    #Now, let's create Year, Month and Day
    date_forecast['Year'] = date_forecast.index.year
    date_forecast['Month'] = date_forecast.index.month
    date_forecast['Day'] = date_forecast.index.day
    
    return date_forecast

#--------------------------------------------------------------------------------------------------#

#Return a list of DataIndex
def create_date_range(dates, range_type, range_length):
    """
    DESCRIPTION
      This function returns a list with DataIndex elements considering the lengh and type of range
    ARGUMENTS
      dates: The original date list
      range_type: For example Days 'D'
      range_lenght: How many new DateIndex considering the type
    RETURN
      A list with DataIndex ranges
    """    
    
    return [(np.array(date, dtype=np.datetime64) + pd.to_timedelta(np.arange(range_length), range_type)) for date in dates]

#--------------------------------------------------------------------------------------------------#

#Return the Dataframe for training with all features, seasonal, exogenous, lagged and target
def prepare_data(data, column, lags):
    """
    DESCRIPTION
      This function returns a specific data model for training, it has all needed features and target
    ARGUMENTS
      df: This is the original DataFrame where pick the data
      column: This is the target variable
      lags: list of number of lgas
    RETURN
      A new DataFrame ready for training
    """  
    #All lagged features
    data_lag = get_lag_features(data, column, lags)
    
    #All new season features needed also for exogenous calculation
    data_se = prepare_seasonal_data(data)
    
    data_model = pd.concat([data_se, data_lag, data[column]], axis= 1)
    data_model.dropna(inplace= True)
    
    return data_model

#--------------------------------------------------------------------------------------------------#

#Similar to prepare_data but for forecasting purposes
def prepare_data_forecast(data, column, lags, period):
    """
    DESCRIPTION
      This function returns a specific data model for forecasting. It has all features but target is unknow
    ARGUMENTS
      df: This is the original DataFrame where pick the data
      column: This is the target variable
      lags: list of number of lgas
      period: Number of T+period to forecast
    RETURN
      A new DataFrame ready for forecasting
    """  
    
    #All lagged features for forecasting - Period ?
    data_lag_forecast = get_lag_features_forecast(data, 'Revenue', lags, period)
    
    #All new season features needed also for exogenous calculation
    seasonal_forecast = get_dataindex_forecast(data, period)
    #All new season features needed also for exogenous calculation
    data_se_forecast = prepare_seasonal_data(seasonal_forecast)
    
    #Data Set Ready for Forecasting
    data_forecast = pd.concat([data_se_forecast, data_lag_forecast], axis= 1)
    
    return data_forecast

#--------------------------------------------------------------------------------------------------#

#This function is not needed-----------------------------------------------------------------------#
#This could be the main function to return the final DataFrame with all manipulations
def dataframe_preparation(df):
    """
    DESCRIPTION
      This function returns a new DataFrame with all Columns well sorted and normalized
    ARGUMENTS
      df: This is original DataFrame to manipulate
    RETURN
      A new DataFrame with Xs Values: Year, Week, Black Friday, Cyber Monday, Direct, SEO, SEM, Affiliate, Email, Display and Y Value: Revenue
    """
    
    #Clone the current DataFrame
    data = df.copy()
    #First of all manipulate the Date column and split into Year, Month, Day and Year-Week
    data = split_dates_by_iso_columns(data, 'Date')
    
    #Let's keep the columns needed
    #Because we are going to use ISO Week and ISO Year, we are not going to use Month
    data = data[['Year', 'Week', 'Program', 'Visits', 'Revenue']]
    
    #We are not going to work with all Programs, just SEO, Direct, SEM, Display, Affiliate and Email
    #We are going to sum in the Display bucket Paid Social, Organic Social, Lenovo Social and Social
    #And the rest will go to the Direct bucket
    data = group_programs(data)
    
    #Now let's build the Xs - Features. Program for us is similar to Traffic Sources
    program_data = data.groupby(['Year', 'Week', 'Program'])['Visits'].sum().unstack().fillna(value = 0).reset_index()
    #In this phase we need to add two categorial variables - Black Friday and Cyber Monday in this DataFrame
    bf_cm_data = get_black_friday_cyber_monday(program_data.iloc[:,0:2])
    
    #Now let's build the Y - Revenue. We are using as primary key for merging Year and Week
    revenue_data = data.groupby(['Year', 'Week'])['Revenue'].sum().reset_index()
    
    #DataFrame is ready
    return pd.concat([bf_cm_data, 
                      program_data[['Direct', 'SEO', 'SEM', 'Affiliate', 'Email', 'Display']], 
                      revenue_data['Revenue']], axis= 1)

#--------------------------------------------------------------------------------------------------#

#---------------------------------FUNCTIONS FOR ARIMA MODELS PURPOSE-------------------------------#

#Get the Train and Test with the better lags considering a maximum of iterations for ARIMA model
def arma(df, column, max_lags):
    """
    DESCRIPTION
      This function creates an ARIMA model
    ARGUMENTS
      df: This is the DataFrame from get train and test data
      column: This is the target variable
      max_lags: Numer of maximun iteration to find the best lag
    RETURN
      A pair of DataFrames, Train and Test with Target Variable and Target Predicted using best lag
    """
    
    #Best Model
    best_RMSE = 100000000000
    best_p = -1
    best_data_train = pd.DataFrame()
    best_data_test = pd.DataFrame()

    for i in range(1, max_lags):
        #Prepare the data adding the lags values
        data = arma_model(df, column, i)
        #Split Train and Test
        X_train, y_train, X_test, y_test = arma_preparation_split(data)
        #Train model
        data_train, data_test = arma_train (X_train, y_train, X_test, y_test, column)
        
        #Get error
        RMSE = arma_rmse(data_train, column)
        
        #If it is better get the better RMSE and lag
        if(RMSE < best_RMSE):
            best_RMSE = RMSE
            best_p = i
            best_data_train = data_train
            best_data_test = data_test
    
    #Show best RMSE and lag
    print(f'Best RMSE:{RMSE} and lags:{best_p}')
    
    #Return Train and Test
    return best_data_train, best_data_test

#--------------------------------------------------------------------------------------------------#

#Create the lagged variables for a specific target
def arma_model(df, column, lags):
    """
    DESCRIPTION
      This function creates a new DataFrame with the lagged variables
    ARGUMENTS
      df: This is the DataFrame from get train and test data
      column: This is the target variable
      lags: Numer of lags
    RETURN
      A new DataFrame with the lagged variables
    """
    
    data = df[[column]].copy()
    
    #Generating the lagged terms
    for i in range(1, lags + 1):
        data[f'{column}-{i}'] = data[column].shift(i)
    
    #Dropna after the lagging
    data.dropna(inplace= True)
    #Reset Index
    data.reset_index(drop= True)
    
    return data

#--------------------------------------------------------------------------------------------------#

#This method is for manual Arima
#Train and Split for Arima
def arma_preparation_split(df):
    """
    DESCRIPTION
      This function returns the X_train, y_train, X_test and y_test for training
    ARGUMENTS
      df: This is the DataFrame from get train and test data
    RETURN
      X_train DataFrame with the lagged variables, y_train target variable and
      X_test DataFrame with the lagged variables to test the trained model and y_test with the real target value
    """
    
    #A copy of the DataFrame    
    #Important DataFrame structure has the lagged values column 2, 3 and so on
    #Let's sort the DataFrame in a more natural way, first X and the last column y
    data = df[df.columns[::-1]].copy()
    
    #We pick 80% of the sample
    sample_size = len(data)
    train_size = (int)(0.8 * sample_size)
    
    #Split Train and Test
    #In order to keep the same Index, we need to reset index in Test but we will do it in Train as well
    #data_train = pd.DataFrame(data[0:train_size]).reset_index(drop= True)
    #data_test = pd.DataFrame(data[train_size:sample_size]).reset_index(drop= True)
    data_train = pd.DataFrame(data[0:train_size])
    data_test = pd.DataFrame(data[train_size:sample_size])
    
    #Split Train and Test - No trained yet
    X_train, y_train = data_preparation(data_train)
    X_test, y_test = data_preparation(data_test)
    
    #All ready for training in next step
    return X_train, y_train, X_test, y_test

#--------------------------------------------------------------------------------------------------#

def sarimax_preparation_split(df):
    """
    DESCRIPTION
      This function returns the X_train, y_train, X_test and y_test for training
    ARGUMENTS
      df: This is the DataFrame from get train and test data
    RETURN
      X_train DataFrame with the features, y_train target variable and
      X_test DataFrame with the features to test the trained model and y_test with the real target value
    """
    
    #A copy of the DataFrame    
    data = df.copy()
    
    #We pick 90% of the sample
    sample_size = len(data)
    train_size = (int)(0.9 * sample_size)
    
    #Split Train and Test
    data_train = pd.DataFrame(data[0:train_size])
    data_test = pd.DataFrame(data[train_size:sample_size])
    
    #Split Train and Test - No trained yet
    X_train, y_train = data_preparation(data_train)
    X_test, y_test = data_preparation(data_test)
    
    #All ready for training in next step
    return X_train, y_train, X_test, y_test

#--------------------------------------------------------------------------------------------------#

#This function train the ARIMA model
def arma_train(X_train, y_train, X_test, y_test, column):
    """
    DESCRIPTION
      This function returns two DataFrames, data_train with y_real and y_predict and data_test with y_real and y_predict
    ARGUMENTS
      X_train: DataFrame with the lagged variables for training
      y_train: Target variable for training
      X_test: DataFrame with the lagged variables for testing
      y_test: Target variable for testing
      column: This is the target variable
    RETURN
      A pair of DataFrames, Train and Test with Target Variable and Target Predicted
    """
    
    #Let's prepare the returned DataFrames
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()
    
    #Model and training    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    #Coefficients of the Regression
    coefficients  = lr.coef_
    intercept = lr.intercept_
    
    #Create the DataFrame with the y_real and y_predict for train
    data_train[column] = y_train
    data_train[f'{column}_predicted'] = X_train.dot(coefficients) + intercept
    #Create the DataFrame with the y_real and y_predict for test
    data_test[column] = y_test
    data_test[f'{column}_predicted'] = X_test.dot(coefficients) + intercept
    
    #Return train and test
    return data_train, data_test    

#--------------------------------------------------------------------------------------------------#

#This function return the RMSE
def sarimax_rmse(series_real, series_predicted):
    """
    DESCRIPTION
      This function returns the RMSE between y_real and y_predict
    ARGUMENTS
      series_real: This is the real y
      series_predicted: This is the predicted y
    RETURN
      Root mean square error
    """
    return np.sqrt(mean_squared_error(series_real, series_predicted))

#--------------------------------------------------------------------------------------------------#

#SARIMAX best parameters by testing all of the combinations
def gridsearch_sarimax(endog, exog, arima_params, sarima_params, trend_params):
    """
    DESCRIPTION
      This function return a list of best AIC for SARIMAX model
      CAUTION: You can run out memory, so you cannot put high values
      This is a intensive method and a pickle file would be stored for continue in the last iteration
    ARGUMENTS
      endog: target values
      exog: X features
      arima_params: This is a dictionary with p, d, q range values. If you put 1, range means 0-1. The d value should be 2 up most
      sarima_params: This is a dictionaty with P, D, Q and Season. Season is a list of values. The D value should be 2 up most
      trend_params: This is a list of values ['n','c','t','ct']
    RETURN
      A dictionary with this format key: (p, d, q) - (P, D, Q, Season) - Trend and value: AIC
      A pickle file will be stored in case the gridsearch crash or stopped abruptly - object.pkl
    """
    
    #Pickle file
    pickle_file = 'object.pkl'
    
    #Disctionary to return
    test_aic = dict()
    
    #ARIMA Parameters
    p = np.arange(arima_params['p'][0], arima_params['p'][1]) 
    d = np.arange(arima_params['d'][0], arima_params['d'][1])
    q = np.arange(arima_params['q'][0], arima_params['q'][1])
    
    #SARIMA Parameters
    P = np.arange(sarima_params['P'][0], sarima_params['P'][1])
    D = np.arange(sarima_params['D'][0], sarima_params['D'][1])
    Q = np.arange(sarima_params['Q'][0], sarima_params['Q'][1])
    seasons = sarima_params['S']
    
    #Trend Parameters
    t = trend_params
    
    #Progress Bar
    pbar = tqdm()
    pbar.reset(p.shape[0] * d.shape[0] * q.shape[0] * P.shape[0] * D.shape[0] * Q.shape[0] * len(t) * len(seasons))

    for i in p:
        for j in d:
            for k in q:
                for w in P:
                    for a in D:
                        for s in Q:
                            for x in seasons:
                                for freq in t:
                                    try:
                                        #SARIMAX Model
                                        model = SARIMAX(endog= endog, exog= exog, order= (i, j, k), 
                                                        seasonal_order= (w, a, s, x), trend= freq)
                                        result = model.fit()
                                        #Get AIC Metrics
                                        test_aic[(i, j, k, '-', w, a, s, x, '-', freq)] = result.aic
                                        
                                        #Save the object for each iteration in case crash
                                        object_to_pickle(test_aic, pickle_file)

                                    except:
                                        continue

                                    #Progress Bar
                                    pbar.update()        
    #Progress Bar
    pbar.refresh()
    
    return test_aic

#--------------------------------------------------------------------------------------------------#

#Triple Exponential Smooth best parameters by testing all of the combinations
def gridsearch_tes(endog, trend_params, damped_params, seasonal_params, period_params, box_cox, bias_params):
    """
    DESCRIPTION
      This function return a list of best RMSE for TES model
      CAUTION: You can run out memory, so you cannot put high values
      This is a intensive method and a pickle file would be stored for continue in the last iteration
    ARGUMENTS
      endog: target values
      trend_params: List of values, Additive - add or Multiplicative - mul
      damped_params: True or False
      seasonal_params = List of values, Additive - add or Multiplicative - mul
      period_params = List of periods, like 1, 2, 7, 28
      box_cox = Box-Cox transformation, True or False
      bias_params = True or False
      
    RETURN
      A dictionary with this format key: (tren, damped, season, period, - box_cox, bias) and value: RMSE
      A pickle file will be stored in case the gridsearch crash or stopped abruptly - object.pkl
    """
    
    #Pickle file
    pickle_file = 'object.pkl'
    
    #Disctionary to return
    test_rmse = dict()
    
    #Progress Bar
    pbar = tqdm()
    pbar.reset(len(trend_params) * len(damped_params) * len(seasonal_params) * len(period_params) *
              len(box_cox) * len(bias_params))
    
    for i in trend_params:
        for j in damped_params:
            for k in seasonal_params:
                for w in period_params:
                    for a in box_cox:
                        for s in bias_params:
                            try:
                                #Triple Exponential Smooth Model
                                model = ExponentialSmoothing(endog= endog, trend= i, damped= j, seasonal= k, 
                                        seasonal_periods= w)
                                result = model.fit(optimized= True, use_boxcox= a, remove_bias= s)
                                #Get RMSE Metrics
                                test_rmse[(i, j, k, w, '-', a, s)] = \
                                    np.sqrt(mean_squared_error(endog, 
                                        result.predict(start= endog.index[0], end= endog.index[-1])))

                                #Save the object for each iteration in case crash
                                object_to_pickle(test_rmse, pickle_file)

                            except:
                                continue
                            
                            #Progress Bar
                            pbar.update()        
    
    #Progress Bar
    pbar.refresh()
    
    return test_rmse

#--------------------------------------------------------------------------------------------------#

#------------------------------FUNCTIONS FOR MACHINE LEARNING PURPOSE------------------------------#

#Variance features analysis
def non_zero_variance(df, columns):
    """
    DESCRIPTION
      This function analyze the standard deviation for each feature
    ARGUMENTS
      df: This is the DataFrame to analyze
      columns: A list of columns with the features
    RETURN
      What feature has non-zero and zero variance
    """
    
    selector = VarianceThreshold()
    selector.fit_transform(columns)
    
    non_zero = set(df.columns[selector.get_support(indices=True)])
    zero = set(df.columns) - set(list([df.columns[-1]])) - non_zero
    
    for i in non_zero:
        print(f'{i} feature: has non-zero variance - Std:{np.std(df[i]):.2f}')
    for i in zero:
        print(f'{i} feature: has zero variance and should be removed  - Std:{np.std(df[i]):-2f}')

#--------------------------------------------------------------------------------------------------#

#Return the X and y for splitting
def data_preparation(df):
    """
    DESCRIPTION
      This function returns a X Pandas set for splitting or predict and y Series for splitting or predict
    ARGUMENTS
      df: This is the DataFrame from get train and test data
    RETURN
      A new DataFrame with Xs features and y target
      IMPORTANT: DataFrame columns order should be <all features><target> this mean, last column is the target
    """
    
    X = df.columns[0:-1] # pandas DataFrame
    y = df.columns[-1] # pandas Series
    
    return df[X], df[y]

#--------------------------------------------------------------------------------------------------#

#Return the Xs and ys for test and train
def data_split(X, y):
    """
    DESCRIPTION
      This function returns the set for training and test
    ARGUMENTS
      X: It is a DataFrame with the features
      y: It is a Series with the objective
    RETURN
      The X Train and Test and y Train and Test
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 1)
    return X_train, X_test, y_train, y_test

#--------------------------------------------------------------------------------------------------#

#Return scaled X_train and X_test based on three scaler Norm, Robust and Power
def data_normalization(X_train, X_test, scale_param):
    """
    DESCRIPTION
      This function X_train and X_test set normalized based on scale method
    ARGUMENTS
      X_train: Train set to scale
      X_test: Test set to scale
      scale_param: Type of normalization:
       'norm': StandarScaler
       'robust': RobustScaler
       'power': PowerTransformer
       'minmax: MinMaxScaler
    RETURN
      X_train scaled
      X_test scale
    """

    #Get the scaler
    scaler = get_scaler(scale_param)
    
    #Train the scaler with the X_train data
    scaler.fit(X_train)
    
    #Scale both sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

#--------------------------------------------------------------------------------------------------#

#Return a scaler
def get_scaler(scale_param):
    """
    DESCRIPTION
      This function returns one of the available scaler among Standard, Robust or Power
    ARGUMENTS
      scale_param: Type of normalization:
       'norm': StandarScaler
       'robust': RobustScaler
       'power': PowerTransformer
    RETURN
      A scaler, Standard, Robust or Power
    """
    
    #Dictionaries of scaling methods
    methods = {'norm': StandardScaler(),
               'robust': RobustScaler(),
               'power': PowerTransformer(),
               'minmax': MinMaxScaler()}
    #Data Normalization
    scaler = methods[scale_param]
    
    return scaler

#Return a model to train with the best hyperparameters discovered previously
def get_ml_model(model):
    """
    DESCRIPTION
      This function returns one of the available machine learning Gradient Boost or XGB
    ARGUMENTS
      model: Type of regressor:
       'gradient': Gradient Boost
       'xgb': XGB       
    RETURN
      A machine learning
    """
    
    #Dictionaries of Machine Learning methods
    models_dict = {'ridge': Ridge(alpha=0.001, fit_intercept=False, solver='sag'),
                'gradient': GradientBoostingRegressor(max_features='auto'),
               'xgb': XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=13,
              min_child_weight=1, missing=np.nan, monotone_constraints='()',
              n_estimators=1000, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)}
    
    return models_dict[model]

#--------------------------------------------------------------------------------------------------#

#Return a DataFrame with the Real, Predict Target and Residuals
def predict_model(model, X, y):
    """
    DESCRIPTION
      This function applies the features to the model and get the prediction
    ARGUMENTS
      X: Features
      y: Target
    RETURN
      A DataFrame with y_real, y_predicted and Residuals
    """
    #Predict this model with Train Data
    y_predict = model.predict(X)
    y_predict = pd.Series(y_predict, index= y.index)
    y_predict = y_predict.rename(f'{y.name}_predicted')
    #Residuals
    forecast = pd.DataFrame([y, y_predict]).T
    forecast['Residuals'] = forecast[y.name] - forecast[f'{y.name}_predicted']
    
    #Create the new DataFrame with the Train y values and Train predict values
    return forecast

#--------------------------------------------------------------------------------------------------#

#Return the Xs and ys for the first fold of one Time Series
def time_series_train_test_split(df, lags):
    """
    DESCRIPTION
      This function returns the set for training and test for Time Series for the first fold
    ARGUMENTS
      df: DataFrame where to apply train and test split
      lags: List with the lag features
    RETURN
      The X Train and Test and y Train and Test
    """
    
    #Get the random split, it will return a list of pair tain and test k-folds
    #We will keep the first one always
    train_index_list, test_index_list = time_series_fold(df)
    train_index = train_index_list[0]
    test_index = test_index_list[0]
    
    return time_series_split(df, train_index, test_index, lags)

#--------------------------------------------------------------------------------------------------#

#This function split the train and test returning the indexes
def time_series_fold(df, fold= 10):
    """
    DESCRIPTION
      This function returns a list of indexes for train, test and split or cross validation
    ARGUMENTS
      df: DataFrame where to apply train and test split
      fold: Number of splits, minumun 2. By default we are picking the 90%/10% which is fold= 10
    RETURN
      Two list, one with all Train_index and Test_index combinations
    """
    #We are going to use K-fold but we are just picking first value
    #K-fold cross-validator
    kf = KFold(fold, shuffle=True)

    #list of train and test indexes
    train_index_list = list()
    test_index_list = list()

    #Train/Test cross-validation splits
    for train_index, test_index in kf.split(df):
        train_index_list.append(train_index)
        test_index_list.append(test_index)
        
    return train_index_list, test_index_list

#--------------------------------------------------------------------------------------------------#

#Return the Xs and ys for test and train considering the Time Series Restrictions
#https://robjhyndman.com/hyndsight/tscv/ thanks Sebas ;-)
def time_series_split(df, train_index, test_index, lags):
    """
    DESCRIPTION
      This function returns the set for training and test for Time Series
    ARGUMENTS
      df: DataFrame where to apply train and test split
      train_index: This is the train index from a k-fold method
      test_index: This is the test index from a k-fold method
      lags: List with the lag features
    RETURN
      The X Train and Test and y Train and Test
    """
    
    #We are going to remove from Train all dependencies with Test
    #If we Y depends on Y-1, Y-2 and so on, we are going to remove those lag variables from Train
    #Basically add the lad to the test index and remove it from the train
    set_test_index = set()
    set_train_index = set(train_index)
    
    for i in lags:
        set_test_index.update(set(test_index + i))
    
    #New train index without test dependencies - IMPORTANT! New set is much smaller than the original one
    new_train_index = list(set_train_index - set_test_index)
    
    #New split for Time Series - Using the new indexes
    X, y = data_preparation(df)
    X_train = X.iloc[new_train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[new_train_index]
    y_test = y.iloc[test_index]
    
    return X_train, X_test, y_train, y_test

#--------------------------------------------------------------------------------------------------#

#This function return the list all fit and predict for Train and Test for Cross-Validation Score or Grid-Search
def cross_val_time_series(model, df, lags, fold, scaler= 'robust'):
    """
    DESCRIPTION
      Execute all train and test for each fold and return a list of Daframe with pair target and predicted
    ARGUMENTS
      model: This is the estimator to fit the data
      df: DataFrame with the Time Series
      lags: List with the lagged features
      fold: Number of splits, minumun 2. By default we are picking the 90%/10% which is fold= 10
      scaler: Type of normalization - By default RobustScaler
       'norm': StandarScaler
       'robust': RobustScaler
       'power': PowerTransformer
    RETURN
      A list K-fold of DataFrames with all target and predict columns
    """
        
    #Provides K-fold train/test indices to split data in train/test sets for Time Series
    train_index_list, test_index_list = time_series_fold(df, fold= fold)
    
    #List of all fit and predict for Train and Test
    forecasts = list()
    
    for train, test in zip(train_index_list, test_index_list):
        #Train and Test Split for each K-fold
        X_train, X_test, y_train, y_test = time_series_split(df, train, test, lags)
        
        #Scale the features
        X_train_scaled, X_test_scaled = data_normalization(X_train, X_test, scaler)
        
        #Fit model with Train
        model.fit(X_train_scaled, y_train)
        #Predict model with Test and measure
        y_predict = model.predict(X_test_scaled)
        y_predict = pd.Series(y_predict, index= y_test.index)
        y_predict = y_predict.rename(f'{y_test.name}_predicted')
        #Create the new DataFrame with the Test y values and Test predict values
        forecasts.append(pd.DataFrame([y_test, y_predict]).T)
        
    return forecasts

#--------------------------------------------------------------------------------------------------#

#This function calculate the metric for each K-fold with Train and Test
def time_series_score(forecasts, column, metric):
    """
    DESCRIPTION
      Evaluate a score for each K-fold Train and Test
    ARGUMENTS
      forecasts: A list K-fold of DataFrames with all target and predict columns
      column: Target column name
      metric: It could be 'rmse', 'bias', 'variance'
    RETURN
      Array of scores of the estimator for each run of the cross validation
    """
    
    #list of metric results
    metrics = list()
    
    for forecast in forecasts:
        #Add the metric in the metrics list
        if metric == 'rmse':
            metrics.append(metric_rmse(forecast, column))
        elif metric == 'bias':
            metrics.append(bias_variance(forecast, column, 'bias'))
        elif metric == 'variance':
            metrics.append(bias_variance(forecast, column, 'variance'))
        
    return np.array(metrics)

#--------------------------------------------------------------------------------------------------#

#This function performs the Cross Validation Test but for Time Series
def cross_val_time_series_score(model, df, column, lags, fold, metric, scaler= 'robust'):
    """
    DESCRIPTION
      Evaluate a score by cross-validation
    ARGUMENTS
      model: This is the estimator to fit the data
      df: DataFrame with the Time Series
      column: Target column name
      lags: List with the lagged features
      fold: Number of splits, minumun 2. By default we are picking the 90%/10% which is fold= 10
      metric: It could be 'rmse', 'bias', 'variance'
      scaler: Type of normalization - By default RobustScaler
       'norm': StandarScaler
       'robust': RobustScaler
       'power': PowerTransformer
    RETURN
      Array of scores of the estimator for each run of the cross validation
    """
    
    #List of all fit and predict for Train and Test
    forecasts = cross_val_time_series(model, df, lags, fold, scaler)
    
    #A list of all metrics for each K-fold
    metrics = time_series_score(forecasts, column, metric)
    
    #Return the metric for each K-fold
    return np.array(metrics)

#--------------------------------------------------------------------------------------------------#

#Exhaustive search over specified parameter values for an estimator in a Time Series
def grid_search_times_series(regressor, regressor_params, df, column, lags, fold, metric, scaler= 'robust'):
    """
    DESCRIPTION
      Evaluate all hyperparameters for a Machine Learning model
    ARGUMENTS
      regressor: This is the estimator to fit the data
      regressor_params: All the hyperparameters to try for a specific model
      df: DataFrame with the Time Series
      column: Target column name
      lags: List with the lagged features
      fold: Number of splits, minumun 2. By default we are picking the 90%/10% which is fold= 10
      metric: It could be 'rmse', 'bias', 'variance'
      scaler: Type of normalization - By default RobustScaler
       'norm': StandarScaler
       'robust': RobustScaler
       'power': PowerTransformer
    RETURN
      models_list: A list with the model with all tested parameters
      metrics_list: A list with all different metric values per model tested
      forecasts_list: A list with the trained model with the target and predicted value in case we want to check Residuals
    """
    
    #Get all parameters combination to test
    parameters = it.product(*(regressor_params[i] for i in regressor_params))
    #IMPORTANT - we need to repeat the combination because len + list for tqmd leaves empty the variable - consume the value
    combination_length = it.product(*(regressor_params[i] for i in regressor_params))
    
    #List to return
    forecasts_list = list()
    metrics_list = list()
    models_list = list()
    
    #Progress Bar
    pbar = tqdm()
    pbar.reset(len(list(combination_length)))
    
    for i in parameters:
        
        #Get the model with the hyperparameters
        model = get_model(regressor, i)
        
        #List of all fit and predict for Train and Test
        forecasts = cross_val_time_series(model, df, lags, fold, scaler)
        
        #Array of all metrics
        metrics = time_series_score(forecasts, column, metric)
        
        #Add the model, forecasts and metrics in the list to return
        models_list.append(model)
        forecasts_list.append(forecasts)        
        metrics_list.append(metrics)
        
        #Progress Bar
        pbar.update()  
        
    #Progress Bar
    pbar.refresh()
    
    return models_list, metrics_list, forecasts_list

#--------------------------------------------------------------------------------------------------#

#Return a specific model with their respective hyperparameters
def get_model(model, params):
    """
    DESCRIPTION
      Construct a specific regressor model with their respective hyperparameters
    ARGUMENTS
      model: This is the estimator to create with its respective parameters
      params: A specific list of hyperparameters
    RETURN
      An instance of the model with the desired parameters
    """       
        
    if model == 'linear':
        return LinearRegression()
    elif model == 'ridge':
        return Ridge(alpha= params[0], fit_intercept= params[1], solver= params[2])
    elif model == 'elastic': 
        return ElasticNet(alpha= params[0], l1_ratio= params[1], fit_intercept= params[2])
    elif model == 'kneighbors':
        return KNeighborsRegressor(n_neighbors= params[0], weights= params[1], algorithm= params[2])
    elif model == 'random':
        return RandomForestRegressor(n_estimators= params[0], max_depth= params[1],
                min_samples_split= params[2], min_samples_leaf= params[3], 
                max_features= params[4], bootstrap= params[5])
    elif model == 'gradient':
        return GradientBoostingRegressor(loss= params[0], n_estimators= params[1],
                min_samples_split= params[2], min_samples_leaf= params[3],
                max_depth= params[4], max_features= params[5])
    elif model == 'xgb':
        return XGBRegressor(n_estimators= params[0], max_depth= params[1], booster= params[2])
    elif model == 'lgbm':
        return LGBMRegressor(num_leaves= params[0], n_estimators= params[1])

#--------------------------------------------------------------------------------------------------#

#Get the best score and hyperparameters for a specific GridSearch
def best_model_metric(models, metrics):
    """
    DESCRIPTION
      Return the best metric(RMSE normally) and best hyperparameters
    ARGUMENTS
      models: List of all tested models
      metrics: List of all metrics for all tebest_metric    RETURN
      best_metric: Best metric after the GridSearch
      best_hyperparameters: Best parameters combination with the lowest metric
    """       
    
    #Best score index
    best_score = pd.DataFrame(metrics).T.mean().sort_values().index[0]
    #Best metric
    best_metric = pd.DataFrame(metrics).T.mean().sort_values()[best_score]
    #Best model
    best_hyperparameters = models[best_score]
    
    return best_metric, best_hyperparameters    
    
#--------------------------------------------------------------------------------------------------#

#Predict the future base on previous data
def forecast_predict(data, column, lags, period, ml_model, scale_param= 'robust', confidence= '95%'):
    """
    DESCRIPTION
      This returns the a forecast and the intervals of confidence based on a period
    ARGUMENTS
      data: DataFrame with the original Time Series
      column: Target column name
      lags: List with the lagged features
      period: Number of days to predict
      ml_model: Machine Learning to select, onlye two options:
      'xgb': XGB
      'gradient': Gradient Boost
      'ridge': Ridge
      scaler: Type of normalization - By default RobustScaler
       'norm': StandarScaler
       'robust': RobustScaler
       'power': PowerTransformer
      confidence: For confidence intervals creation
    RETURN
      A DataFrame with the predicted values and confidence intervals
    """
    
    #Convert list lags into arrays lags for manipulation
    lags_a = np.array(lags)
    #Prepare the scaler
    scaler = get_scaler(scale_param)
    #Forecast to return
    forecast = pd.DataFrame()
    #Confidence intervals
    intervals = pd.DataFrame()
    #Get the machine learning to train and use for forecasting
    model = get_ml_model(ml_model)
    #Index for the forecast - period would be same as lags
    index_range = create_date_range([data.index[-1] + pd.to_timedelta(1, 'D')], 'D', period)
    
    #Need to prepare the data for forecasting. We need to merge the set for prediction because there are
    #Some dummies which are not available. Mainly the seasonal and exogenous features
    data_forecast = prepare_data_forecast(data, column, lags, period)
    seasonal_forecast = prepare_seasonal_data(data)
    #Need to concatenate by rows and then by columns
    temp_forecast = pd.concat([seasonal_forecast, data_forecast])
    data_forecast = pd.concat([temp_forecast, data[column]], axis= 1).fillna(0, downcast= 'infer')
    
    #Therefore, we will have the forecast with all features needed otherwise, there will be missing some
    #due to we use categorical ones for dates
    
    #For each lag, we need to create a new model with the new lagged features
    for i in range(period):
        #Let's start from i=0 means lag T-lags list -1, -2, and so on
        #from i=1 means lag T-lags array -1-1=T-2, -2-1=T-3 and so on 
        lag = lags_a + i
        
        #--Begin Training the model
        #Prepare the data for training
        data_model = prepare_data(data, column, lag)
        
        #Train and Test Split. However, we are not going to use Test
        X_train, X_test, y_train, y_test = time_series_train_test_split(data_model, lag)
        
        #Train the scaler with the X_train data
        scaler.fit(X_train)
        #Scale just Train
        X_train_scaled = scaler.transform(X_train)
        
        #Apply the fit to regressor
        model.fit(X_train_scaled, y_train)
        
        #Revenue, Predicted and Residuals. This is for creating the Confidence Intervals
        predict_train = predict_model(model, X_train_scaled, y_train)
        #Calculate the Standard Deviation
        std = predict_train['Residuals'].std()
        ##--End Training the model
        
        #Start with the prediction
        #Pick the <i> position value to forecast and so on and apply the scaler
        X_forecast = data_forecast.loc[index_range[0][i]][:-1]
        X_forecast = np.array(X_forecast).reshape(1, -1)
        X_forecast_scaled = scaler.transform(X_forecast)
        #Predict
        y_forecast = model.predict(X_forecast_scaled)
        
        #Prepare the Series with the Index=Date and Column=Revenue_forecast
        #y_forecast[0] because it is an array
        y_forecast_serie = pd.Series(y_forecast[0], index= [index_range[0][i]])
        y_forecast_serie = y_forecast_serie.rename(f'{column}_forecast')
        
        #Now prepare the Confidence Intervals
        lower = y_forecast - get_confidence(confidence) * std
        upper = y_forecast + get_confidence(confidence) * std
        
        y_lower_serie = pd.Series(lower, index= [index_range[0][i]])
        y_lower_serie = y_lower_serie.rename(f'{column}_lower')
        
        y_upper_serie = pd.Series(upper, index= [index_range[0][i]])
        y_upper_serie = y_upper_serie.rename(f'{column}_upper')
        
        #Add the Series to the final Forecast DataFrame
        forecast = pd.concat([forecast, y_forecast_serie.to_frame()])
        #Add the Confidence Intervals
        intervals = pd.concat([intervals, pd.concat([y_lower_serie.to_frame(), y_upper_serie.to_frame()], axis= 1)])
    
    
    #Return the final DataFrame
    return pd.concat([forecast, intervals], axis= 1)

#--------------------------------------------------------------------------------------------------#

#--------------------------------FUNCTIONS FOR DEEP LEARNING PURPOSE-------------------------------#

#Return a list of RNNs
def get_rnn_list(period):
    """
    DESCRIPTION
      This functions create a list of RNN based on the number of periods to predict in a forecasting
    ARGUMENTS
      period: Number of days to predict
    RETURN
      A list of RNN
    """
  
    rnn_list = list()
    
    for i in range(period):
        #RNN
        rnn = models.Sequential()
        #Three layers - Let's pick shape= 61 x 2, then 61 x1 and 30 and 7 nodes
        rnn.add(layers.Dense(122, input_shape=(61,), activation='relu')) 
        rnn.add(layers.Dense(61, activation='relu'))
        rnn.add(layers.Dense(30, activation='relu'))
        rnn.add(layers.Dense(7, activation='relu'))
        rnn.add(layers.Dense(1))
        #Compile the RNN
        rnn.compile(optimizer= optimizers.RMSprop(), loss= losses.mean_squared_error, metrics= [metrics.mean_squared_error]) 
        #Add the RNN to the list
        rnn_list.append(rnn)
    
    return rnn_list

#--------------------------------------------------------------------------------------------------#

#Predict the future base on previous data for RNN
def forecast_predict_rnn(data, column, lags, period, model, epochs, scale_param= 'robust', confidence= '95%'):
    """
    DESCRIPTION
      This returns the a forecast and the intervals of confidence based on a period
    ARGUMENTS
      data: DataFrame with the original Time Series
      column: Target column name
      lags: List with the lagged features
      period: Number of days to predict
      model: A list of RNN with the same size as period
      epochs: Number of iterations
      'xgb': XGB
      'gradient': Gradient Boost
      scaler: Type of normalization - By default RobustScaler
       'norm': StandarScaler
       'robust': RobustScaler
       'power': PowerTransformer
      confidence: For confidence intervals creation
    RETURN
      A DataFrame with the predicted values and confidence intervals
    """
    
    #Convert list lags into arrays lags for manipulation
    lags_a = np.array(lags)
    #Prepare the scaler
    scaler = get_scaler('robust')
    #Forecast to return
    forecast = pd.DataFrame()
    #Confidence intervals
    intervals = pd.DataFrame()
    #Index for the forecast - period would be same as lags
    index_range = create_date_range([data.index[-1] + pd.to_timedelta(1, 'D')], 'D', period)


    #Need to prepare the data for forecasting. We need to merge the set for prediction because there are
    #Some dummies which are not available. Mainly the seasonal and exogenous features
    data_forecast = prepare_data_forecast(data, column, lags, period)
    seasonal_forecast = prepare_seasonal_data(data)
    #Need to concatenate by rows and then by columns
    temp_forecast = pd.concat([seasonal_forecast, data_forecast])
    data_forecast = pd.concat([temp_forecast, data[column]], axis= 1).fillna(0, downcast= 'infer')

    #Therefore, we will have the forecast with all features needed otherwise, there will be missing some
    #due to we use categorical ones for dates

    #For each lag, we need to create a new model with the new lagged features
    for i in range(period):
        #Let's start from i=0 means lag T-lags list -1, -2, and so on
        #from i=1 means lag T-lags array -1-1=T-2, -2-1=T-3 and so on 
        lag = lags_a + i

        #--Begin Training the model
        #Prepare the data for training
        data_model = prepare_data(data, column, lag)

        #Train and Test Split. However, we are not going to use Test
        X_train, X_test, y_train, y_test = time_series_train_test_split(data_model, lag)

        #Train the scaler with the X_train data
        scaler.fit(X_train)
        #Scale just Train
        X_train_scaled = scaler.transform(X_train)

        #Apply the fit to the specific neuronal network
        model[i].fit(X_train_scaled, y_train, epochs= epochs)

        y_predict_train = model[i].predict(X_train_scaled)
        y_predict_train = pd.DataFrame(y_predict_train, index= y_train.index, columns= [f'{y_train.name}_predicted'])
        predict_train = pd.concat([y_train.to_frame(), y_predict_train], axis= 1)
        predict_train['Residuals'] = predict_train[column] - predict_train[f'{column}_predicted']

        #Calculate the Standard Deviation
        std = predict_train['Residuals'].std()

        #Start with the prediction
        #Pick the <i> position value to forecast and so on and apply the scaler
        X_forecast = data_forecast.loc[index_range[0][i]][:-1]
        X_forecast = np.array(X_forecast).reshape(1, -1)
        X_forecast_scaled = scaler.transform(X_forecast)
        
        #Predict
        y_forecast = model[i].predict(X_forecast_scaled)
        
        #Prepare the Series with the Index=Date and Column=Revenue_forecast
        #y_forecast[0] because it is an array
        y_forecast_serie = pd.Series(y_forecast[0], index= [index_range[0][i]])
        y_forecast_serie = y_forecast_serie.rename(f'{column}_forecast')

        #Now prepare the Confidence Intervals
        lower = y_forecast - get_confidence(confidence) * std
        upper = y_forecast + get_confidence(confidence) * std

        y_lower_serie = pd.Series(lower[0], index= [index_range[0][i]])
        y_lower_serie = y_lower_serie.rename(f'{column}_lower')

        y_upper_serie = pd.Series(upper[0], index= [index_range[0][i]])
        y_upper_serie = y_upper_serie.rename(f'{column}_upper')

        #Add the Series to the final Forecast DataFrame
        forecast = pd.concat([forecast, y_forecast_serie.to_frame()])
        #Add the Confidence Intervals
        intervals = pd.concat([intervals, pd.concat([y_lower_serie.to_frame(), y_upper_serie.to_frame()], axis= 1)])

    return pd.concat([forecast, intervals], axis= 1)

#--------------------------------------------------------------------------------------------------#

#------------------------------FUNCTIONS FOR ANALYSIS/METRICS PURPOSE------------------------------#

#Augmented Dickey-Fuller Test
def test_adf(df, column):
    """
    DESCRIPTION
      This function calculate the Augmented Dickey-Fuller Test to proof
      H0 if p-value > 0.05 or T-Test is bigger than critical values interval - No stationary
      H1 if p-value < 0.05 or T-Test is smaller than critical values interval - Stationary
    ARGUMENTS
      df: This is the DataFrame with the data
      column: Variable to apply the test
    RETURN
      If it is a stationary or not variable
    """
    
    interval = list(['1%', '5%', '10%'])
    greater_smaller = lambda x, y: '<' if x < y else '>'
    check = lambda x, y: 'Stationary' if x < y else 'No Stationary'
    h0 = False
    
    adf = adfuller(df[column])
    t_test = adf[0]
    p_value = adf[1]
    t_interval = adf[4]
    
    print('** Augmented Dickey-Fuller Test **\n')
    for i in t_interval:
        print(f'T-test: {t_test} {greater_smaller(t_test, t_interval[i])} Confidence Interval[{i}]: {t_interval[i]} - Result: {check(t_test, t_interval[i])}')
    
    print(f'\nP-Value: {p_value} {greater_smaller(p_value, 0.05)} 0.05 - Result: {check(p_value, 0.05)}')  

#--------------------------------------------------------------------------------------------------#

#This function return the RMSE
def metric_rmse(df, column):
    """
    DESCRIPTION
      This function returns the RMSE between y_real and y_predict
    ARGUMENTS
      df: This is the DataFrame where to calculate the RMSE
      column: Column name of the objective
    RETURN
      Root mean square error
    """
    return np.sqrt(mean_squared_error(df[column], df[f'{column}_predicted']))

#--------------------------------------------------------------------------------------------------#

#This is a metric for measuring the bias and variance (tradeoff)
def bias_variance(df, column, tradeoff):
    """
    DESCRIPTION
      This function returns the Bias or Variance for a model
    ARGUMENTS      
      df: This is the DataFrame where to calculate the Bias or Variance
      column: Column name of the objective
      tradeoff: What type of metric to apply in the model
    RETURN
      Bias or Variance
    """
        
    #Bias or Variance metric
    if tradeoff == 'variance':
        return np.corrcoef(df[column], df[f'{column}_predicted'])[0][1]
    elif tradeoff == 'bias':
        return np.mean(df[column] - df[f'{column}_predicted'])

#--------------------------------------------------------------------------------------------------#

#--------------------------------FUNCTIONS FOR SERIALIZING PURPOSE---------------------------------#

#Serializing
def object_to_pickle(pkl_object, pkl_file):
    """
    DESCRIPTION
      This function saves the object into a pickle file
    ARGUMENTS
      pkl_object: Object to serialize
      pkl_file: Pickle file name
    RETURN
      Nothing, it will create a pickle file
    """
    
    pkl = open(pkl_file, 'wb')
    pickle.dump(pkl_object, pkl)                     
    pkl.close()

#--------------------------------------------------------------------------------------------------#
    
#De-serializing
def pickle_to_object(pkl_file):
    """
    DESCRIPTION
      This function retrieves the pickle object
    ARGUMENTS
      pkl_file: Pickle file name
    RETURN
      De-serialize object
    """    
    
    pkl = open(pkl_file, 'rb')     
    pkl_object = pickle.load(pkl)
    pkl.close()
    return pkl_object

#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#

    
