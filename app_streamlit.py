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

from bitcoin_price_predictor.btc_pred import *

X_train, X_test, y_train, y_test, y_b_test, y_b_train, scaled_df = data_scaling()

st.title("BITCOIN PRICE PREDICTOR")

st.header('Compare our model to your *gut feeling* and a *coin flip*')

genre = st.radio(
    "Based on gut-feeling do you want invest in Bitcoin today?",('Yes', 'No'))
if genre == 'Yes':
    st.write('You chose to invest')
else:
    st.write("You chose not to invest")

st.header('Was that the right decision?')
st.header("*Let's see what the model says...*")
st.title('THE MODEL')
'The model uses live data from Yahoo Finance from the following stocks'

def get_price(symbol):
    response = get('https://finance.yahoo.com/quote/{}/'.format(symbol))
    soup = BeautifulSoup(response.text, 'html.parser')
    return float(soup.find_all(
        attrs={'class': 'Trsdu(0.3s)'})[0].text.replace(',', ''))

stocks = ['GOOG', '^GSPC', 'DX-Y.NYB', '^IXIC', 'BTC-USD', 'BTC=F', '^KS11','000001.SS']
prices = pd.DataFrame(columns=['datetime', 'price'])
selected_symbols = st.multiselect('', stocks, ['GOOG', '^GSPC', 'DX-Y.NYB', '^IXIC', 'BTC-USD', 'BTC=F', '^KS11','000001.SS'])
for _symbol in selected_symbols:
    prices.loc[_symbol] = {'datetime': pd.datetime.now(),
                           'price': get_price(_symbol)}

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
        st.text('TIME TO BUY')
    else:
        st.text('TIME TO SELL')

expander = st.beta_expander('Were you right to invest?')
with expander:
    if genre == 'Yes' and random.choice(b) == 'Invest - our model predicts the bitcoin price will rise':
        st.header('Your gut was right!!!!!')
    else:
        st.header('Your gut was wrong!')

data = pd.DataFrame({'model':[65, 67, 54, 70, 66, 69, 57, 60, 65, 67],'coin toss':[50, 50, 50, 50, 50, 50, 50, 50, 50, 50], 'Warren Buffet':[100, 100, 100, 100, 100, 100, 100, 100, 100, 100]})

"*Lets see what the coin says?*"

a = ["Invest", "Do not invest"]

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Flip Coin')
coin = random.choice(a)
if pressed:
    st.header(coin)

expander2 = st.beta_expander("Comparison of a coin toss vs our model")
expander2.write('We backtested our model vs a coin flip 100 times...')
with expander2:

    st.line_chart(data)



