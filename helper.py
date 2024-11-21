import pandas as pd

def create_date_range(start='2018-08-31', end='2024-08-31', freq='M', periods=None):
    """
    Create a DatetimeIndex with enhanced flexibility
    
    Parameters:
    -----------
    start : str, datetime, optional
        Start date of the range (default: '2018-08-31')
    end : str, datetime, optional
        End date of the range (default: '2024-08-31')
    freq : str, optional
        Frequency of the date range (default: 'M' for month end)
        Common frequencies:
        - 'D': calendar day
        - 'B': business day
        - 'W': weekly
        - 'M': month end
        - 'Q': quarter end
        - 'Y': year end
    periods : int, optional
        Number of periods to generate if end is not specified
    
    Returns:
    --------
    pd.DatetimeIndex
        A DatetimeIndex with specified parameters
    """
    try:
        date_range = pd.date_range(
            start=start, 
            end=end, 
            freq=freq, 
            periods=periods
        )
        return date_range
    except Exception as e:
        print(f"Error creating date range: {e}")
        return None