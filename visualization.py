
#Streamlit, Altair and Image
import streamlit as st
import altair as alt
from PIL import Image
#Array and Frames Libraries
import pandas as pd
import numpy as np
#Date Libraries
from datetime import datetime
from dateutil.relativedelta import relativedelta
#Functions
from functions import *

#Constants
model_type_list = ['Select', 'Classical', 'Machine Learning', 'Deep Learning']
classical_model_list = ['Select', 'SARIMAX', 'Holt-Winters']
ml_model_list = ['Select', 'Ridge', 'Gradient Boost', 'XGB']
rnn_model_list = ['Select', 'RNN']
predict_days = [2, 3, 4, 5, 6, 7, 14, 28]

#Starting and Welcome to the App
def starting_welcome():
    #Read icon
    icon = Image.open('./icon.png')
    
    c1, c2 = st.beta_columns((1,1))
    c1.title('Welcome to the Forecast Sale Model')
    c2.image(icon)

    st.subheader('This project has been defined to estimate the future sales in a specific window time frame and using multiple \
    machine learnings models')

    st.markdown('We are trying to resolve a Time Series problem where uncertanty is quite difficult to forecast. However, this would be\
    our starting point')

    st.markdown(
    """    
    Steps in order to forecast a Time Series:   
    * Select the file to parse. It has to be in *.xlsx format and having these columns
        * Date: When the record took place
        * Program: This is the type of traffic. It could be empty but the column has to exist
        * Visits: This is the number of visitors. It could be empty but the column has to exist
        * Revenue: This is the amount of money made that day. In general this the target of our model
    """)


#Loading Sidebar
def sidebar():

    st.sidebar.title("Forecast Sale Model")
    st.sidebar.subheader("Please select a file")
    uploaded_file = st.sidebar.file_uploader("Select a file .xlsx", type="xlsx")
    
    if not uploaded_file:
        st.stop()
    else:
        return uploaded_file


#Loading file
@st.cache
def loading_data(file):
    
    if not file:
        #Nothing
        st.stop()
    elif file:
        #Load the main data
        data = pd.read_excel(file)
        #Clean zeros
        data = fill_zero_revenue(data)
        #Manipulate the DataFrame a little bit. Modifying the Dates
        data_small = split_dates_by_columns(data, 'Date')
        #For this early stage I discard type of Traffic and just sum all of them and group by Index, Year, Month and Day
        data_small = data_small.groupby(['Date', 'Year','Month','Day'])[['Visits','Revenue']].sum()
        #Reset Index after having made a groupby
        data_small = data_small.reset_index(['Year', 'Month', 'Day'])
        #Resample to D and 00h00min00sec
        data_small = data_small.resample('D', offset='00h00min00sec').sum()

    return data_small


#Draw a chart
def starting_chart(data):
    
    #chart = alt.Chart(data).mark_line().encode(x= ('Day'), y= ('Revenue'))
    chart = alt.Chart(data.reset_index()).mark_line().encode(alt.X('Date'), alt.Y('Revenue')
            ).properties(width= 800, height= 500, title= 'Revenue Evolution')
    st.altair_chart(chart)

    
#Loading Sidebar
def sidebar_model(data):

    #Constant Section
    lags = [1, 2, 3, 4, 5, 6, 7, 364]
    column = 'Revenue'
    
    st.sidebar.subheader("Please choose a Forecast Model")
    #Select the Model
    type_model = st.sidebar.selectbox('Model Type', model_type_list)
    if type_model == 'Select':
        
        st.warning('Please choose a Model Type')
    
    elif type_model == 'Classical':
        
        classical_model = st.sidebar.selectbox('Classical Type', classical_model_list)
        if classical_model == 'Select':
            
            st.warning('Please choose a Classical Type')
        elif classical_model == 'SARIMAX':
            
            st.markdown('SARIMAX Model')
            data_exogenous, start_date, end_date = slider_date_classical(data, classical_model)
            result_sarimax = sarimax_model(data_exogenous, start_date, end_date)
            date_forecast, number_days = slider_date_classical_forecast(end_date)
            sarimax_forecast_model(result_sarimax, date_forecast, number_days)
        
        elif classical_model == 'Holt-Winters':
            
            st.markdown('Holt-Winters')
            data_exogenous, start_date, end_date = slider_date_classical(data, classical_model)
            result_holt = holt_model(data_exogenous, start_date, end_date)
            number_days = slider_date_classical_holt(end_date)
            holt_forecast_model(result_holt, number_days)
            
    elif type_model == 'Machine Learning':
        
        ml_model = st.sidebar.selectbox('Machine Learning Type', ml_model_list)
        if ml_model == 'Select':
            
            st.warning('Please choose a Machine Learning Type')
        elif ml_model == 'Ridge':

            #Ridge Model
            st.markdown('Ridge Model')
            model= Ridge(alpha=0.001, fit_intercept=False, solver='sag')
            draw_chart_ml(model, data, column, lags)
            draw_forecast_ml(data, lags, 'ridge')

        elif ml_model == 'Gradient Boost':
            
            st.markdown('Gradient Boost Model')
            model = GradientBoostingRegressor(max_features='auto')
            draw_chart_ml(model, data, column, lags)
            draw_forecast_ml(data, lags, 'gradient')
        
        elif ml_model == 'XGB':
            
            st.markdown('XGB Model')  
            model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=13,
              min_child_weight=1, missing=np.nan, monotone_constraints='()',
              n_estimators=1000, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
            draw_chart_ml(model, data, column, lags)
            draw_forecast_ml(data, lags, 'xgb')
            
    elif type_model == 'Deep Learning':
        
        rnn_model = st.sidebar.selectbox('Deep Learning Type', rnn_model_list)
        if rnn_model == 'Select':
            st.warning('Please choose a Deep Learning Type')
        else:
            st.markdown('RNN Model')
            draw_chart_rnn(data, column, lags)


def draw_chart_rnn(data, column, lags):   

    data_model = prepare_data(data, column, lags)
    #Train and Test Split
    X_train, X_test, y_train, y_test = time_series_train_test_split(data_model, lags)
    #Scale the features
    X_train_scaled, X_test_scaled = data_normalization(X_train, X_test, 'robust')
    
    #RNN
    rnn = models.Sequential()
    #Three layers - Let's pick shape= 61 x 2, then 61 x1 and 30 and 7 nodes
    rnn.add(layers.Dense(122, input_shape=(X_train_scaled.shape[1],), activation='relu')) 
    rnn.add(layers.Dense(61, activation='relu'))
    rnn.add(layers.Dense(30, activation='relu'))
    rnn.add(layers.Dense(7, activation='relu'))
    rnn.add(layers.Dense(1))
    #Compile the RNN
    rnn.compile(optimizer= optimizers.RMSprop(), loss= losses.mean_squared_error, metrics= [metrics.mean_squared_error]) 
    
    #Train the RNN
    rnn.fit(X_train_scaled, y_train, epochs= 500)
    
    #Let's predict this model with Train Data
    y_predict_train = rnn.predict(X_train_scaled)
    y_predict_train = pd.DataFrame(y_predict_train, index= y_train.index, columns= [f'{y_train.name}_predicted'])
    #Create the new DataFrame with the Train y values and Train predict values
    forecast_train = pd.concat([y_train.to_frame(), y_predict_train], axis= 1)
    
    chart = alt.Chart(forecast_train.reset_index()).transform_fold(
        [x for x in forecast_train.columns],
        ).mark_line().encode(x='Date:T', y='value:Q', color='key:N'
        ).properties(width= 800, height= 500, title= 'Real vs. Predicted')

    st.altair_chart(chart)
    
    rmse = 'RMSE: ' + str(metric_rmse(forecast_train.dropna(), 'Revenue'))
    st.write(rmse)

def draw_chart_ml(model, data, column, lags):
    
    data_model = prepare_data(data, column, lags)
    X_train, X_test, y_train, y_test = time_series_train_test_split(data_model, lags)
    #Scale the features
    X_train_scaled, X_test_scaled = data_normalization(X_train, X_test, 'robust')
    model.fit(X_train_scaled, y_train)
    
    forecast = predict_model(model, X_train_scaled, y_train)
    forecast_reduced = forecast[['Revenue', 'Revenue_predicted']]
    
    chart = alt.Chart(forecast_reduced.reset_index()).transform_fold(
        [x for x in forecast_reduced.columns],
        ).mark_line().encode(x='Date:T', y='value:Q', color='key:N'
        ).properties(width= 800, height= 500, title= 'Real vs. Predicted')

    st.altair_chart(chart)
    
    rmse = 'RMSE: ' + str(metric_rmse(forecast_reduced.dropna(), 'Revenue'))
    st.write(rmse)

def draw_forecast_ml(data, lags, type_ml):
    
    st.sidebar.subheader("Please choose a data range for prediction")
    
    #Dates range, minimun one week so, end date is last date - 7 days
    end_date = datetime.strptime(str(data.index[-1]), '%Y-%m-%d %H:%M:%S')
    starting_start = end_date
    starting_end = starting_start
    starting_date = st.sidebar.date_input("Chose starting date", 
            value= starting_start, min_value= starting_start, max_value= starting_end)
    #Same as above, we need minimun one week
    number_days = st.sidebar.selectbox('How many days?', predict_days)
    
    forecast = forecast_predict(data, 'Revenue', lags, number_days, type_ml, 'robust', '95%')
    
    if type_ml == 'ridge':
        title = 'Ridge - Revenue Forecast'
    elif type_ml == 'gradient':
        title = 'Gradient Boost - Revenue Forecast'
    elif type_ml == 'xgb':
        title = 'XGB - Revenue Forecast'
    
    
    chart = alt.Chart(forecast.reset_index()).mark_line().encode(alt.X('index'), alt.Y('Revenue_forecast')
            ).properties(width= 800, height= 500, title= title)
    
    confidence = alt.Chart(forecast.reset_index()).mark_area(opacity=0.5
        ).encode(x='index', y='Revenue_lower', y2='Revenue_upper')
    
    st.altair_chart(chart + confidence)

    
            
def holt_model(data, start_date, end_date):

    #Convert datetime
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    
    X_train, y_train, X_test, y_test = sarimax_preparation_split(data)
    
    if X_train.shape[0] < 49:
        seasonal_periods = 7
    else:
        seasonal_periods = 49
    
    model = ExponentialSmoothing(endog= y_train, trend= 'mul', damped= False, seasonal= 'mul', 
                seasonal_periods= seasonal_periods)
    result = model.fit(optimized= True, use_boxcox= True, remove_bias= True)
    
    #Predict taking full Train
    forecast = result.predict(start= start_date, end= end_date)
    forecast.rename(f'{y_train.name}_predicted', inplace= True)
    
    data_train_predict = pd.concat([y_train, forecast], axis= 1)
    #Show only the range
    data_train_predict = data_train_predict.loc[start_date:end_date]
    
    chart = alt.Chart(data_train_predict.reset_index()).transform_fold(
        [x for x in data_train_predict.columns],).mark_line().encode(x='Date:T', y='value:Q', color='key:N'
        ).properties(width= 800, height= 500, title= 'Real vs. Predicted')
    
    st.altair_chart(chart)
    
    rmse = 'RMSE: ' + str(metric_rmse(data_train_predict.dropna(), 'Revenue'))
    st.write(rmse)
    
    return result
            
#Sarimax Model
def sarimax_model(data_exogenous, start_date, end_date):
    
    #Convert datetime
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    
    X_train, y_train, X_test, y_test = sarimax_preparation_split(data_exogenous)
    
    #Fit considering best hyperparameters
    model = SARIMAX(endog= y_train, exog= X_train[['Black_Friday', 'Easter', 'Covid']],
                order= (3, 0, 0), seasonal_order= (1, 0, 1, 7), trend= 't')
    result = model.fit()
    
    #Predict taking full Train
    forecast = result.get_prediction(start= start_date, end= end_date, 
            exog= data_exogenous.loc[start_date:end_date][['Black_Friday', 'Easter', 'Covid']])
    mean_forecast = forecast.predicted_mean
    #Rename mean_forecast column name
    mean_forecast.rename(f'{y_train.name}_predicted', inplace= True)
    
    data_train_predict = pd.concat([data_exogenous[[data_exogenous.columns[-1]]], mean_forecast], axis= 1)
    #Show only the range
    data_train_predict = data_train_predict.loc[start_date:end_date]
    
    confidence_intervals = forecast.conf_int()

    chart = alt.Chart(data_train_predict.reset_index()).transform_fold(
        [x for x in data_train_predict.columns],).mark_line().encode(x='Date:T', y='value:Q', color='key:N'
        ).properties(width= 800, height= 500, title= 'Real vs. Predicted')
    
    confidence = alt.Chart(confidence_intervals.reset_index()).mark_area(opacity=0.5
        ).encode(x='Date', y='lower Revenue', y2='upper Revenue')
    
    st.altair_chart(chart + confidence)
    
    rmse = 'RMSE: ' + str(metric_rmse(data_train_predict.dropna(), 'Revenue'))
    st.write(rmse)
    
    return result

#This creates the data range for training the model
def slider_date_classical(data, type_model):
    
    if type_model == 'SARIMAX':
        data_exogenous = get_exogenous_features(data)
        X_train, y_train, X_test, y_test = sarimax_preparation_split(data_exogenous)
    elif type_model == 'Holt-Winters':
        data_exogenous = data
        X_train, y_train, X_test, y_test = sarimax_preparation_split(data)
    
    start = datetime.strptime(str(X_train.index[0]), '%Y-%m-%d %H:%M:%S') 
    end = datetime.strptime(str(X_train.index[-1]), '%Y-%m-%d %H:%M:%S')
    
    #Dates range, minimun one week so, end date is last date - 7 days
    starting_start = start
    starting_end = end - relativedelta(days= 7)
    starting_date = st.sidebar.date_input("Chose starting date for training", 
            value= starting_start, min_value= starting_start, max_value= starting_end)
    #Same as above, we need minimun one week
    ending_start = starting_date + relativedelta(days= 365)
    ending_end = end 
    ending_date = st.sidebar.date_input("Chose ending date for training", 
            value= ending_start, min_value= ending_start, max_value= ending_end)
    
    return data_exogenous, starting_date, ending_date

def slider_date_classical_holt(end_date):

    st.sidebar.subheader("Please choose a data range for prediction")
    
    #Dates range, minimun one week so, end date is last date - 7 days
    starting_start = end_date
    starting_end = starting_start + relativedelta(days= 90)
    starting_date = st.sidebar.date_input("Chose starting date", 
            value= starting_start, min_value= starting_start, max_value= starting_end)
    #Same as above, we need minimun one week
    number_days = st.sidebar.selectbox('How many days?', predict_days)
    
    return number_days


    
    
#This prepare the data for forecasting in Sarimax
def slider_date_classical_forecast(end_date):

    st.sidebar.subheader("Please choose a data range for prediction")
    
    #Dates range, minimun one week so, end date is last date - 7 days
    starting_start = end_date
    starting_end = starting_start
    starting_date = st.sidebar.date_input("Chose starting date", 
            value= starting_start, min_value= starting_start, max_value= starting_end)
    #Same as above, we need minimun one week
    number_days = st.sidebar.selectbox('How many days?', predict_days)
    
    from_date = pd.to_datetime(starting_date.strftime('%Y-%m-%d'))
    index_range = create_date_range([from_date], 'D', number_days)
    
    #Create new rows, index
    row_list = list()
    for i in index_range[0]:
        row_list.append(pd.Series(name= i))
    
    date_forecast = pd.DataFrame(row_list)
    
    #Now, let's create Year, Month and Day
    date_forecast['Year'] = date_forecast.index.year
    date_forecast['Month'] = date_forecast.index.month
    date_forecast['Day'] = date_forecast.index.day
    
    return date_forecast, number_days


def holt_forecast_model(result_holt, number_days):
    
    mean_forecast = result_holt.forecast(number_days)
    mean_forecast.rename('Revenue_forecast', inplace= True)
    mean_forecast = pd.DataFrame(mean_forecast)
    
    chart = alt.Chart(mean_forecast.reset_index()).mark_line().encode(alt.X('index'), alt.Y('Revenue_forecast')
        ).properties(width= 800, height= 500, title= 'Holt-Winters - Revenue Forecast')
    
    st.altair_chart(chart)


#This function make the forecast
def sarimax_forecast_model(result, data, steps):
    
    data_exogenous = get_exogenous_features(data)
    
    forecast = result.get_forecast(steps= steps, exog= data_exogenous[['Black_Friday', 'Easter', 'Covid']])
    mean_forecast = forecast.predicted_mean
    mean_forecast.rename('Revenue_forecast', inplace= True)
    mean_forecast = pd.DataFrame(mean_forecast)
    
    confidence_intervals = forecast.conf_int()

    chart = alt.Chart(mean_forecast.reset_index()).mark_line().encode(alt.X('index'), alt.Y('Revenue_forecast')
            ).properties(width= 800, height= 500, title= 'SARIMAX - Revenue Forecast')
    
    confidence = alt.Chart(confidence_intervals.reset_index()).mark_area(opacity=0.5
        ).encode(x='index', y='lower Revenue', y2='upper Revenue')
    
    st.altair_chart(chart + confidence)

starting_welcome()
file = sidebar()
data = loading_data(file)
starting_chart(data)
sidebar_model(data)


