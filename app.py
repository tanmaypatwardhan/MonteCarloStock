import streamlit as st
from streamlit_tags import st_tags, st_tags_sidebar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import yfinance as yf
from scipy.stats import norm

def getData(stocks, num_shares, start = "2010-01-01"):
    data = pd.DataFrame()
    for i in range(0, len(stocks)):
        data[i] = yf.download(stocks[i], start)['Adj Close'] * num_shares[i]
    
    result = np.sum(data, axis = 1)
    return result


def computeReturns(data, iterations, days):
    log_return = np.log(1 + data.pct_change())
    mu = log_return.mean()
    var = log_return.var()
    drift = (mu - (0.5 * var))
    std_dev = log_return.std()

    result = np.exp(drift + std_dev * norm.rvs(size = (days, iterations)))
    
    prices = np.zeros_like(result)
    prices[0] = data.iloc[-1]
    for i in range(1, days):
        prices[i] = prices[i - 1] * result[i]

    x = pd.DataFrame(prices).iloc[-1]
    
    graph = sns.histplot(x, stat="density", kde=True)
    graph.lines[0].set_color('crimson')
    
    #for i in range(prices.shape[1]):  
        #plt.plot(prices[:, i], lw=1)

    avg_price = prices[-1].sum()/iterations
    percent_change = (avg_price - prices[0][0])/avg_price * 100
    avg_profit = (prices[-1] - prices[0]).sum()/iterations
    
    print(f'Percent change: {percent_change}%')
    print(f'Average profit: ${avg_profit}')
    print(f'Average value: ${avg_price}')



with st.sidebar:
    df = pd.DataFrame(
    [
       {"Stock": "GOOG", "# Shares": 5, "Include": True},
       {"Stock": "RTX", "# Shares": 10, "Include": True}
       
    ]
    )
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width = True)
    stocks = edited_df.loc[edited_df["Include"] == True, "Stock"].astype("string").tolist()
    shares = edited_df.loc[edited_df["Include"] == True, "# Shares"].astype("float").tolist()

    conf = st.button("Confirm", use_container_width = True, type = "primary")
    
        

if(conf):
    x = computeReturns(getData(stocks, shares), 10000, 365)
    #fig = plt.figure(figsize=(15, 14))

    sns.histplot(x, stat="density", kde=True)

    st.pyplot(plt.gcf(), use_container_width = True)

    