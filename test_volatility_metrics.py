import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path to import functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from daily_returns.py
from volatility_metrics import (
    calculate_daily_returns,
    get_return_std_dev,
    get_rolling_volatility,
    calculate_atr,
    calculate_volatility_ratio,
    calculate_bollinger_bands,
    calculate_bollinger_bands_width
)


@pytest.fixture
def sample_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    close_prices = [base_price]
    
    for ret in returns[1:]:
        close_prices.append(close_prices[-1] * (1 + ret))
    
    close_prices = np.array(close_prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    open_prices = close_prices + np.random.normal(0, 0.005, 100) * close_prices
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })

@pytest.fixture
def simple_data():
    """Create simple test data for basic testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
        'open': [100, 101, 102, 98, 99, 103, 105, 104, 106, 108],
        'high': [102, 103, 104, 100, 101, 105, 107, 106, 108, 110],
        'low': [99, 100, 101, 97, 98, 102, 104, 103, 105, 107],
        'close': [101, 102, 103, 99, 100, 104, 106, 105, 107, 109]
    })


class TestVolatilityMetrics:
    """Test suite for volatility metrics functions."""

    def test_calculate_daily_returns_basic(self, simple_data):
        """Test basic daily returns calculation."""
        returns = calculate_daily_returns(simple_data)
        
        # Check return type
        assert isinstance(returns, pd.Series)
        
        # Check length
        assert len(returns) == len(simple_data)
        
        # First return should be NaN
        assert pd.isna(returns.iloc[0])
        
        # Check specific calculations
        expected_return_1 = (102 - 101) / 101  # (102-101)/101
        assert abs(returns.iloc[1] - expected_return_1) < 1e-10
        
        expected_return_2 = (103 - 102) / 102  # (103-102)/102
        assert abs(returns.iloc[2] - expected_return_2) < 1e-10

    def test_calculate_daily_returns_edge_cases(self):
        """Test edge cases for daily returns."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame({'close': []})
        returns = calculate_daily_returns(empty_df)
        assert len(returns) == 0
        
        # Test with single row
        single_row = pd.DataFrame({'close': [100]})
        returns = calculate_daily_returns(single_row)
        assert len(returns) == 1
        assert pd.isna(returns.iloc[0])
        
        # Test with zero prices (should handle division by zero)
        zero_prices = pd.DataFrame({'close': [0, 100, 0]})
        returns = calculate_daily_returns(zero_prices)
        assert pd.isna(returns.iloc[0])  # First is always NaN
        assert np.isinf(returns.iloc[1])  # 100/0 = inf

    def test_get_return_std_dev(self, sample_data):
        """Test standard deviation calculation."""
        returns = calculate_daily_returns(sample_data)
        std_dev = get_return_std_dev(returns)
        
        # Check return type
        assert isinstance(std_dev, float)
        
        # Check it's positive
        assert std_dev > 0
        
        # Check precision (rounded to 4 decimal places)
        assert len(str(std_dev).split('.')[-1]) <= 4
        
        # Compare with pandas std (should be close)
        expected_std = round(returns.std(), 4)
        assert std_dev == expected_std

    def test_get_return_std_dev_edge_cases(self):
        """Test edge cases for standard deviation."""
        # Test with constant returns
        constant_returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        std_dev = get_return_std_dev(constant_returns)
        assert std_dev == 0.0
        
        # Test with single value
        single_return = pd.Series([0.01])
        std_dev = get_return_std_dev(single_return)
        assert pd.isna(std_dev) or std_dev == 0.0

    def test_get_rolling_volatility(self, sample_data):
        """Test rolling volatility calculation."""
        returns = calculate_daily_returns(sample_data)
        window = 10
        rolling_vol = get_rolling_volatility(returns, window)
        
        # Check return type
        assert isinstance(rolling_vol, pd.Series)
        
        # Check length
        assert len(rolling_vol) == len(returns)
        
        # First (window-1) values should be NaN
        assert all(pd.isna(rolling_vol.iloc[:window-1]))
        
        # Check that non-NaN values are positive
        valid_values = rolling_vol.dropna()
        assert all(valid_values >= 0)

    def test_calculate_atr_basic(self, simple_data):
        """Test basic ATR calculation."""
        atr = calculate_atr(simple_data, window=3)
        
        # Check return type
        assert isinstance(atr, pd.Series)
        
        # Check length
        assert len(atr) == len(simple_data)
        
        # First few values should be NaN due to rolling window
        assert pd.isna(atr.iloc[0])
        assert pd.isna(atr.iloc[1])
        
        # ATR values should be positive where not NaN
        valid_atr = atr.dropna()
        assert all(valid_atr > 0)

    def test_calculate_atr_edge_cases(self, simple_data):
        """Test ATR edge cases."""
        # Test with window larger than data
        atr = calculate_atr(simple_data, window=20)
        assert all(pd.isna(atr))
        
        # Test with window of 1
        atr = calculate_atr(simple_data, window=1)
        assert not all(pd.isna(atr))

    def test_calculate_volatility_ratio(self, sample_data):
        """Test volatility ratio calculation."""
        vol_ratio = calculate_volatility_ratio(sample_data, recent_window=10, long_window=20)
        
        # Check return type
        assert isinstance(vol_ratio, pd.Series)
        
        # Check length
        assert len(vol_ratio) == len(sample_data)
        
        # Values should be positive where not NaN
        valid_ratios = vol_ratio.dropna()
        assert all(valid_ratios > 0)

    def test_calculate_bollinger_bands(self, simple_data):
        """Test Bollinger Bands calculation."""
        bb = calculate_bollinger_bands(simple_data, window=5, num_std_dev=2)  # Updated parameter name
        
        # Check return type
        assert isinstance(bb, pd.DataFrame)
        
        # Check columns
        expected_columns = ['middle_band', 'upper_band', 'lower_band']
        assert all(col in bb.columns for col in expected_columns)
        
        # Check length
        assert len(bb) == len(simple_data)
        
        # Upper band should be greater than middle band
        valid_data = bb.dropna()
        if not valid_data.empty:
            assert all(valid_data['upper_band'] >= valid_data['middle_band'])
            assert all(valid_data['middle_band'] >= valid_data['lower_band'])

    def test_calculate_bollinger_bands_edge_cases(self):
        """Test Bollinger Bands edge cases."""
        # Test with insufficient data
        small_data = pd.DataFrame({
            'close': [100, 101, 102]
        })
        bb = calculate_bollinger_bands(small_data, window=5, num_std_dev=2)  # Updated parameter name
        assert len(bb) == 3
        # Most values should be NaN due to insufficient window
        assert bb.isna().sum().sum() > 0

    def test_calculate_bollinger_bands_width(self, simple_data):
        """Test Bollinger Bands width calculation."""
        width = calculate_bollinger_bands_width(simple_data, window=5, num_std_dev=2)  # Updated parameter name
        
        # Check return type
        assert isinstance(width, pd.Series)
        
        # Check length
        assert len(width) == len(simple_data)
        
        # Width should be positive where not NaN
        valid_width = width.dropna()
        assert all(valid_width > 0)

    def test_integration_workflow(self, sample_data):
        """Test a complete workflow using multiple functions."""
        # Calculate daily returns
        returns = calculate_daily_returns(sample_data)
        
        # Calculate standard deviation
        std_dev = get_return_std_dev(returns)
        
        # Calculate rolling volatility
        rolling_vol = get_rolling_volatility(returns, window=10)
        
        # Calculate Bollinger Bands
        bb = calculate_bollinger_bands(sample_data, window=20, num_std_dev=2)  # Updated parameter name
        
        # All should complete without errors
        assert returns is not None
        assert std_dev is not None
        assert rolling_vol is not None
        assert bb is not None
        
        # Check data consistency
        assert len(returns) == len(sample_data)
        assert len(rolling_vol) == len(sample_data)
        assert len(bb) == len(sample_data)


# Additional utility tests
class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_missing_columns(self):
        """Test behavior when required columns are missing."""
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='D'),
            'close': [100, 101, 102, 103, 104]
            # Missing 'high' and 'low' columns
        })
        
        # This should work for functions that only need 'close'
        returns = calculate_daily_returns(incomplete_data)
        assert len(returns) == 5
        
        # This should fail for ATR (needs high, low, close)
        with pytest.raises(KeyError):
            calculate_atr(incomplete_data, window=3)

    def test_invalid_parameters(self, sample_data):
        """Test behavior with invalid parameters."""
        returns = calculate_daily_returns(sample_data)
        
        # Test negative window - pandas raises ValueError for negative windows
        with pytest.raises(ValueError):
            get_rolling_volatility(returns, window=-1)
        
        # Test zero window - pandas allows this but returns all NaN
        zero_window_result = get_rolling_volatility(returns, window=0)
        assert all(pd.isna(zero_window_result))
        
        # Test with None window - pandas raises ValueError for non-integer windows
        with pytest.raises(ValueError):
            get_rolling_volatility(returns, window=None)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
