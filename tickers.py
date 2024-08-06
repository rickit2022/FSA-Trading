import yfinance as yf
import pandas as pd
from utils import saveFrame
from datetime import datetime

"""
File for fetching closing price data from Yahoo Finance
"""

def getClose(tickers, start, end, interval='1d') -> pd.DataFrame:
    data = yf.download(' '.join(tickers), start, end, interval=interval)
    return data

if __name__ == "__main__":
    """
    List of tickers:
    Apple, AAPL
    Tesla, TSLA
    Amazon, AMZN
    Microsoft, MSFT
    Google, GOOGL
    """
    tickers = ['AAPL', 'TSLA', 'AMZN', 'MSFT', 'GOOGL']
    start = '2021-01-01' 
    end = '2022-12-31'
    period =f"{datetime.strptime(start, '%Y-%m-%d').strftime('%d.%m.%Y')}--{datetime.strptime(end, '%Y-%m-%d').strftime('%d.%m.%Y')}"

    df = getClose(tickers, start, end)

    for ticker in tickers:
        saveFrame(df, "data/tickersHistory", keys= [f"{ticker}({period})"], overwrite=True, index = True)
        