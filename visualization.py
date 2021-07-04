
#Streamlit, Altair and Image
import streamlit as st
import altair as alt
from PIL import Image
#Array and Frames Libraries
import pandas as pd
import numpy as np
#Date Libraries
from datetime import datetime
#Functions
from functions import *

#Constants
model_type_list = ['Select', 'Classical', 'Machine Learning', 'Deep Learning']
classical_model_list = ['Select', 'SARIMAX']
ml_model_list = ['Select', 'Ridge', 'Gradient Boost', 'XGB']
rnn_model_list = ['Select', 'RNN']

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
@st.cache
def starting_chart(data):
    
    #chart = alt.Chart(data).mark_line().encode(x= ('Day'), y= ('Revenue'))
    chart = alt.Chart(data.reset_index()).mark_line().encode(alt.X('Date'), alt.Y('Revenue')
            ).properties(width= 800, height= 500, title= 'Revenue Evolution')
    st.altair_chart(chart)

    
#Loading Sidebar
@st.cache
def sidebar_model(data):

    st.sidebar.subheader("Please choose a Forecast Model")
    #Select the Model
    type_model = st.sidebar.selectbox('Model Type', model_type_list)
    if type_model == 'Select':
        
        st.warning('Please choose a Model Type')
    
    elif type_model == 'Classical':
        
        classical_model = st.sidebar.selectbox('Classical Type', classical_model_list)
        if classical_model == 'Select':
            st.warning('Please choose a Classical Type')
        else:
            st.markdown('SARIMAX Model')
            result = sarimax_model(data)
    elif type_model == 'Machine Learning':
        
        ml_model = st.sidebar.selectbox('Machine Learning Type', ml_model_list)
        if ml_model == 'Select':
            st.warning('Please choose a Machine Learning Type')
        else:
            st.markdown('ML')
            
    elif type_model == 'Deep Learning':
        
        rnn_model = st.sidebar.selectbox('Deep Learning Type', rnn_model_list)
        if rnn_model == 'Select':
            st.warning('Please choose a Deep Learning Type')
        else:
            st.markdown('RNN')


#Sarimax Model
@st.cache
def sarimax_model(data):
    
    data_exogenous = get_exogenous_features(data)
    X_train, y_train, X_test, y_test = sarimax_preparation_split(data_exogenous)
    #Fit considering best hyperparameters
    model = SARIMAX(endog= y_train, exog= X_train[['Black_Friday', 'Easter', 'Covid']],
                order= (3, 0, 0), seasonal_order= (1, 0, 1, 7), trend= 't')
    return model.fit()

def slider_date(data):
    start_date = datetime.strptime(str(data.index[0]), '%Y-%m-%d %H:%M:%S') 
    end_date = datetime.strptime(str(data.index[-1]), '%Y-%m-%d %H:%M:%S') 
    max_days = end_date - start_date
    format = 'DD MMMM YYYY'
    slider = st.sidebar.slider('Select date', min_value=start_date, value=end_date ,max_value=end_date, format=format)
    





starting_welcome()
file = sidebar()
data = loading_data(file)
starting_chart(data)
#sidebar_model(data)
slider_date(data)

