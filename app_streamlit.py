import streamlit as st
import numpy as np
import pandas as pd
import random
import datetime
from datetime import date
import altair as alt
from requests import get
from bs4 import BeautifulSoup
from tensorflow.keras import models
import os
from PIL import Image

from bitcoin_price_predictor.btc_pred import *

X_train, X_test, y_train, y_test, y_b_test, y_b_train, scaled_df = data_scaling()

#st.markdown('<style>body{background-color: #333333;}</style>',unsafe_allow_html=True)
st.title("BITCOIN PRICE PREDICTOR")
image = Image.open('Imagem1.png').resize((1024,600))
st.image(image, use_column_width= True)


st.header("Are you sure today is a good day to invest?")
"Compare your *gut feeling* with our model prediction"

#btc_image = Image.open('btc_image.png').resize((80,80))
genre = st.radio(
    "Based on gut-feeling do you want to invest in Bitcoin today?",("Yes! I feel I should invest 💰 today.", "No! Maybe another day."))
#if genre == 'Yes':
#    st.write('You chose to invest')
#else:
#    st.write("You chose not to invest")

st.header('Is that the right decision? 🧐')
"*Let's see what our model says...*"
st.title('Our Prediction:')
'This prediction is made through a Neural Network based on Yahoo Finance data.'

st.header('Please choose the stock prices you want to use as features:')

stocks_names=['Google (GOOG)','S&P 500 (^GSPC)','US Technology Index (DX-Y.NYB)','NASDAQ (^IXIC)','Bitcoin Futures (BTC=F)','KOSPI Composite - Korean Stock Index (^KS11)','Shanghai Composite – Chinese Stock Index (000001.SS)']

#if st.checkbox('Show raw data'):
#    st.subheader('Stocks')
#    st.write(stocks_names)



def get_price(symbol):
    response = get('https://finance.yahoo.com/quote/{}/'.format(symbol))
    soup = BeautifulSoup(response.text, 'html.parser')
    return float(soup.find_all(
        attrs={'class': 'Trsdu(0.3s)'})[0].text.replace(',', ''))

#st.line_chart(chart_data)

stocks = ['GOOG', '^GSPC', 'DX-Y.NYB', '^IXIC', 'BTC-USD', 'BTC=F', '^KS11','000001.SS']
prices = pd.DataFrame(columns=['datetime', 'price'])
selected_symbols = st.multiselect(label="Choose among:", options=stocks_names, default=stocks_names)
#for _symbol in stocks:
#    prices.loc[_symbol] = {'datetime': pd.datetime.now(),
#                           'price': get_price(_symbol)}

today = date.today()
st.write('Model input is the date today:', today)
b = ["Invest - our model predicts the bitcoin price will rise", "Do not invest - our model predicts the bitcoin price will fall"]

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Run the model')

if pressed:
    path_to_model = os.path.join(os.path.dirname(__file__), "bitcoin_price_predictor", "data", "model")
    loaded_model = models.load_model(path_to_model)
    today_test_data = scaled_df.iloc[-7:,:-1].values.reshape(1,7,36)
    if loaded_model.predict(today_test_data)[0,0] > 0.5:
        going_up_image = Image.open('istockphoto-801479766-1024x1024.jpg').resize((1024,600))
        st.markdown("<h1 style='text-align: center; color: DodgerBlue;'>TIME TO BUY!</h1>", unsafe_allow_html=True)
        st.image(going_up_image, use_column_width= True)


    else:
        going_down_image = Image.open('istockphoto-503640774-1024x1024.jpg').resize((1024,600))
        st.markdown("<h1 style='text-align: center; color: Crimson;'>TIME TO SELL!</h1>", unsafe_allow_html=True)
        st.image(going_down_image, use_column_width= True)



#expander = st.beta_expander('Were you right to invest?')
#with expander:
#    if genre == 'Yes' and random.choice(b) == 'Invest - our model predicts the bitcoin price will rise':
#        st.header('Your gut was right!!!!!')
#    else:
#        st.header('Your gut was wrong!')

#data = pd.DataFrame({'model':[65, 67, 54, 70, 66, 69, 57, 60, 65, 67],'coin toss':[50, 50, 50, 50, 50, 50, 50, 50, 50, 50], 'Warren Buffet':[100, 100, 100, 100, 100, 100, 100, 100, 100, 100]})

#"*Lets see what the coin says?*"
#
#a = ["Invest", "Do not invest"]
#
#left_column, right_column = st.beta_columns(2)
#pressed = left_column.button('Flip Coin')
#coin = random.choice(a)
#if pressed:
#    st.header(coin)

#expander2 = st.beta_expander("Comparison of a coin toss vs our model")
#expander2.write('We backtested our model vs a coin flip 100 times...')
#with expander2:

    #st.line_chart(data)



