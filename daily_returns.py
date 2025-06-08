import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_excel("daily_bitcoin_ohlc.xlsx")

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

## Date Function

def extract_date_components(timestamp_input) -> dict:
    """
    Transform a timestamp (string in "YYYY-MM-DD HH:MM:SS" format *or* a
    datetime.datetime object) into a dictionary of calendar-based features.

    Parameters
    ----------
    timestamp_input : str | datetime.datetime
        Either the timestamp string (e.g. "2023-10-26 14:30:00")
        or an already-parsed datetime object.

    Returns
    -------
    dict
        {
            "year":               2023,
            "quarter":            4,
            "month_number":       10,
            "month_name":         "October",
            "week_of_year":       43,      # ISO-8601 week number
            "day_of_week_number": 4,       # Monday=1 … Sunday=7  (ISO)
            "day_of_week_name":   "Thursday"
        }
    """

# Checks if timestamp input is of instance/type datetime
    if isinstance(timestamp_input, datetime):
        dt = timestamp_input
    elif isinstance(timestamp_input, str):
        try:   
            dt = datetime.strptime(timestamp_input, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:          
            raise ValueError(
                "Timestamp string must match 'YYYY-MM-DD HH:MM:SS'"
            ) from exc
    else:
        raise TypeError(
            "timestamp_input must be a datetime object or a timestamp string"
        )

    features = {
        "year":               dt.year,
        "quarter":            (dt.month - 1) // 3 + 1,          # 1–4
        "month_number":       dt.month,                         # 1–12
        "month_name":         dt.strftime("%B"),                # "January", ("%B" is the full month name)
        "week_number":         dt.isocalendar().week,         # "January"
        "day_of_week_number": dt.weekday() + 1,                    # 1=Mon … 7=Sun 
        "day_of_week_name":   dt.strftime("%A"),                # "Monday"…, ("%A" is the full day name)
    }

    return features

def create_date_components_df(df) -> pd.DataFrame:
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Apply the function to each timestamp and create a DataFrame from the results
    date_components_df = df['timestamp'].apply(extract_date_components).apply(pd.Series)
    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df, date_components_df], axis=1)
    return df

df=create_date_components_df(df)
print(df.head())
# Save the enhanced DataFrame to a new file
df.to_csv('bitcoin_data_with_date_components.csv', index=False)


# ====================================================================================

def get_return_std_dev(return_series):
    """
    Calculates the standard deviation of returns from a Series of daily returns.    

    Parameters:
        return_series (pd.Series): Series containing daily returns.

    Returns:
        float: The standard deviation of returns, rounded to 4 decimal places.

    Example usage:
        std_dev = get_return_std_dev(daily_returns)
    """
    return round(return_series.std(), 4)


def get_rolling_volatility(return_series, window):
    """
    Calculates the rolling standard deviation of returns over a specified window.

    Parameters:
        return_series (pd.Series): Series containing daily returns.
        window (int): The number of days to calculate the rolling standard deviation over.
    
    Returns:
        pd.Series: A Series containing the rolling standard deviation of returns.

    Example usage:
        rolling_volatility = get_rolling_volatility(daily_returns, 20)
    """
    return return_series.rolling(window).std()  

def calculate_atr(df: pd.DataFrame, window) -> pd.Series:
    """
    Calculate the Average True Range (ATR) indicator.
    
    ATR measures volatility by calculating the average of the True Range over a specified period.
    True Range is the greatest of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns
        window (int): Period for calculating the moving average (default: 14)
    
    Returns:
        pd.Series: The Average True Range values
    
    Example:
        atr = calculate_atr(df, window=14)
    """
    # Extract the required price columns
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate the three components of True Range
    # 1. Current High - Current Low
    high_low = high - low
    
    # 2. Absolute value of (Current High - Previous Close)
    high_prev_close = np.abs(high - close.shift(1))
    
    # 3. Absolute value of (Current Low - Previous Close)
    low_prev_close = np.abs(low - close.shift(1))
    
    # True Range is the maximum of the three values
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Calculate ATR using Simple Moving Average
    # Note: Traditional ATR uses Wilder's smoothing (EMA), but SMA is also common
    atr_sma = true_range.rolling(window=window).mean()
    
    return atr_sma

def calculate_volatility_ratio(df: pd.DataFrame, recent_window: int, long_window: int) -> pd.Series:
    """
    Calculate the volatility ratio of a DataFrame.
    
    The volatility ratio is calculated as the ratio of the standard deviation of the recent returns to the standard deviation of the long-term returns.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' column
        recent_window (int): Window for recent returns
        long_window (int): Window for long-term returns
    
    Returns:            
        pd.Series: The volatility ratio values
    
    Example:
        volatility_ratio = calculate_volatility_ratio(df, recent_window=20, long_window=50)
    """
    # Calculate recent returns  
    recent_atr = calculate_atr(df, recent_window)
    long_atr = calculate_atr(df, long_window)

    # Calculate volatility ratio
    volatility_ratio = recent_atr / long_atr

    return volatility_ratio

def calculate_bollinger_bands(df: pd.DataFrame, window: int, num_std_dev: int) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for a DataFrame.
    
    Bollinger Bands are a technical analysis tool that consists of three lines:
    - Middle Band: Moving average of the closing prices
    - Upper Band: Middle Band + (num_std_dev * standard deviation of the closing prices)
    - Lower Band: Middle Band - (num_std_dev * standard deviation of the closing prices)

    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' column
        window (int): Window for the moving average
        num_std_dev (int): Number of standard deviations to include in the bands

    Returns:
        pd.DataFrame: DataFrame containing the Bollinger Bands

    Example:
        bollinger_bands = calculate_bollinger_bands(df, window=20, num_std_dev=2)  

    """

    closing_prices = df['close']

    middle_band = closing_prices.rolling(window=window).mean()

    rolling_std = closing_prices.rolling(window=window).std()

    upper_band = middle_band + (num_std_dev * rolling_std)

    lower_band = middle_band - (num_std_dev * rolling_std)

    bollinger_bands = pd.DataFrame({
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band
    })

    return bollinger_bands

def calculate_bollinger_bands_width(df: pd.DataFrame, window: int, num_std_dev: int) -> pd.Series:
    """
    Calculate the width of Bollinger Bands.
    
    The width is calculated as the difference between the upper and lower bands divided by the middle band.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' column
        window (int): Window for the moving average
        num_std_dev (int): Number of standard deviations to include in the bands

    Returns:
        pd.Series: The width of the Bollinger Bands                                                                 

    Example:
        bollinger_bands_width = calculate_bollinger_bands_width(df, window=20, num_std_dev=2)
    """
    bollinger_bands = calculate_bollinger_bands(df, window, num_std_dev)

    return (bollinger_bands['upper_band'] - bollinger_bands['lower_band']) / bollinger_bands['middle_band']
