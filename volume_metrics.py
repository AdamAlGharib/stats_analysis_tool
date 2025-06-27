

def volume_change(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the volume change of a DataFrame.
    
    The volume change is calculated as the ratio of the current volume to the previous volume.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'volume' column
    
    Returns:
        pd.Series: The volume change values
    """

    volume_change = df["volume"].pct_change()

    return volume_change



def relative_volume_change(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the relative volume change of a DataFrame.
    
    The relative volume change is calculated as the ratio of the current volume to the previous volume.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'volume' column
        window (int): Window for the moving average
    
    Returns:
        pd.Series: The relative volume change values
    """

    N_day_avg_volume = df["volume"].rolling(window=window).mean()
    current_volume = df["volume"]

    relative_volume_change = current_volume / N_day_avg_volume

    return relative_volume_change
    


def calculate_volume_trend(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the volume trend of a DataFrame.
    
    The volume trend is calculated as slope of linear regression of volume over time.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'volume' column
        window (int): Window for the moving average
    
    Returns:    
        pd.Series: The volume trend values
    """
    
    def calculate_slope(values):
        """Calculate the slope of linear regression for a series of values."""
        if len(values) < 2 or values.isna().any():
            return np.nan
        
        # Create x values (time indices)
        x = np.arange(len(values))
        
        # Fit linear regression (degree 1 polynomial)
        # polyfit returns [slope, intercept]
        coefficients = np.polyfit(x, values, 1)
        
        # Return the slope (first coefficient)
        return coefficients[0]
    
    # Apply rolling window and calculate slope for each window
    volume_trend = df["volume"].rolling(window=window).apply(calculate_slope, raw=False)
    
    return volume_trend
    