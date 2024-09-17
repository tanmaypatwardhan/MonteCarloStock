import streamlit as st
from streamlit_tags import st_tags, st_tags_sidebar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import yfinance as yf
from scipy.stats import norm

def getData(stocks, num_shares, start = "2014-01-01"):
    data = pd.DataFrame()
    for i in range(0, len(stocks)):
        data[i] = yf.download(stocks[i], start)['Adj Close'] * num_shares[i]
    
    result = np.sum(data, axis = 1)
    return result


def computeReturns(data, days, iterations=10000):
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
    #sharpe_ratio = (percent_change - 5)/(x.std()/prices[0][0])
    sharpe_ratio = x.std()/prices[0][0]
    print(f'Percent change: {percent_change}%')
    print(f'Average profit: ${avg_profit}')
    print(f'Average value: ${avg_price}')
    print(f'Sharpe Ratio: {sharpe_ratio}')
    



    return x

"""
def stock_metrics_1(prices, iterations=10000):
    avg_price = prices[-1].sum()/iterations
    percent_change = (avg_price - prices[0][0])/avg_price * 100
    avg_profit = (prices[-1] - prices[0]).sum()/iterations
"""







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
    data = computeReturns(getData(stocks, shares), 365)
    bins = 30
    counts, bin_edges = np.histogram(data, bins = bins)



    hist_data = pd.DataFrame({
        'Portfolio Value': np.round(bin_edges[:-1]),  # Start of each bin
        'Probability': counts/10000
    })
    hist_data.set_index('Portfolio Value', inplace=True)
    st.bar_chart(hist_data, x_label = "Portfolio Value", y = "Probability")

    
    