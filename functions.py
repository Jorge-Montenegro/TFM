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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import GammaRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
#Model Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
#Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
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
from patsy import dmatrices
#Regular Expressions Libraries
import re

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
def draw_autocorrelation(df, mode, lags):
    """
    DESCRIPTION
      This draw an autocorrelation or partial autocorrelation chart based on parameter
    ARGUMENTS
      df: DataFrame where the data is stored
      mode: Type of correlation:
       'auto': Autocorrelation
       'partial': Partial autocorrelation
    RETURN
      One plot
    """

    #Check type of autocorrelation and draw it    
    if mode == 'auto':
        plot_acf(df[df.columns[-1]], lags= lags)
    elif mode == 'partial':
        plot_pacf(df[df.columns[-1]], lags= lags)
    
    #Horizontal lines around 0.2 correlation threshold
    plt.hlines(0.2, 0, lags, color= 'r', linestyles= 'dotted', linewidths= 3, alpha=0.5)
    plt.hlines(-0.2, 0, lags, color= 'r', linestyles= 'dotted', linewidths= 3, alpha=0.5)
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

#This function is not used for it was needed in the past. It is a backtracking function
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
    x = df.index
    #Lets prepare a polynomial regression line for Revenue
    y = df[column]
    polynomial_coef = np.polyfit(x, y, degree)
    
    #Prepare the Axes and size
    fig, ax = plt.subplots(1, 1)
    
    #Draw the plot with the right titles and labels
    ax.set_title(f'{column} and Polynomial Adjustment')
    ax.set_xlabel('Obervations Range') 
    ax.set_ylabel(column)
    plt.plot(x, y)
    plt.plot(x, backtracking_polynomial_parameters(polynomial_coef, len(polynomial_coef), x), c= 'r', linewidth= 3)

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
    
    #Return the merge Dataframes
    return pd.concat([data, df[columns]], axis= 1)

#--------------------------------------------------------------------------------------------------#

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
    data_train = pd.DataFrame(data[0:train_size]).reset_index(drop= True)
    data_test = pd.DataFrame(data[train_size:sample_size]).reset_index(drop= True)
    
    #Split Train and Test - No trained yet
    X_train, y_train = data_preparation(data_train)
    X_test, y_test = data_preparation(data_test)
    
    #All ready for training in next step
    return X_train, y_train, X_test, y_test

#--------------------------------------------------------------------------------------------------#

#This function train the ARIMA model
def arma_train (X_train, y_train, X_test, y_test, column):
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
def arma_rmse(df, column):
    """
    DESCRIPTION
      This function returns the RMSE between y_real and y_predict
    ARGUMENTS
      df: This is the DataFrame where to calculate the RMSE
    RETURN
      Root mean square error
    """
    return np.sqrt(mean_squared_error(df[column], df[f'{column}_predicted']))

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
      A new DataFrame with Xs Values: Year, Week, Black Friday, Cyber Monday, Direct, SEO, SEM, Affiliate, Email, Display
      and Y Value: Revenue
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

#Return a normalized DataFrame based on two scaler Norm and Robust
def data_normalization(df, scale_param):
    """
    DESCRIPTION
      This function returns a new DataFrame with all elements normalized based on scale method
    ARGUMENTS
      df: DataFrame to normalize
      scaler: Type of normalization:
       'norm': StandarScaler
       'robust': RobustScaler
       'power': PowerTransformer
    RETURN
      New DataFrame normalized
    """
    
    #Dictionaries of scaling methods
    methods = {'norm': StandardScaler(),
               'robust': RobustScaler(),
               'power': PowerTransformer()}
    #Data Normalization
    scaler = methods[scale_param]
    columns = df.columns
    
    return pd.DataFrame(scaler.fit_transform(df), columns= columns)

#--------------------------------------------------------------------------------------------------#

#This is a metric for measuring the bias and variance (tradeoff)
def bias_variance(model, X_real, y_real, tradeoff):
    """
    DESCRIPTION
      This function returns the Bias or Variance for a model
    ARGUMENTS
      model: Regression model to apply predict
      X_real: Real features
      y_real: Real objective or dependant variable
      tradeoff: What type of metric to apply in the model
    RETURN
      Bias or Variance
    """
        
    #Predict
    y_pred = model.predict(X_real)
    #Bias or Variance metric
    if tradeoff == 'variance':
        return np.corrcoef(y_pred, y_real)[0][1]
    elif tradeoff == 'bias':
        return np.mean(y_pred - y_real)

#--------------------------------------------------------------------------------------------------#

#-----------------------------------FUNCTIONS FOR ANALYSIS PURPOSE---------------------------------#

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
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#

    
