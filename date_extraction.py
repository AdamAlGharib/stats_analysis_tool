import pandas as pd
from datetime import datetime

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

