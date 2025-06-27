import pandas as pd
from datetime import datetime

df = pd.read_csv("top5_daily_ohlcv_since_2018-02-09.csv")

# print(df.head())

def calculate_daily_returns(df) -> pd.Series:
    """
        Calculate the daily returns from a DataFrame with a 'Close' column.

        Parameters:
            df (pd.DataFrame): DataFrame containing historical price data with a 'Close' column.

        Returns:
            pd.Series: A Series of daily return values (as decimal percentages).
    """

    daily_returns = df['close'].pct_change()
    # .pct_change() is a built in pandas method 
    # ( calculates the percentage change between the current and the prev element)
    return daily_returns

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['Daily Return'] = calculate_daily_returns(df)
