Test summary
✅ Functions Tested:
calculate_daily_returns() - Tests basic calculation and edge cases

get_return_std_dev() - Tests standard deviation calculation with various scenarios

get_rolling_volatility() - Tests rolling window volatility calculations

calculate_atr() - Tests Average True Range calculations

calculate_volatility_ratio() - Tests volatility ratio between different time periods

calculate_bollinger_bands() - Tests Bollinger Bands calculation

calculate_bollinger_bands_width() - Tests Bollinger Bands width calculation

🔧 Issues Fixed in daily_returns.py:
Added missing numpy import - Fixed NameError in ATR calculations
Fixed variable name conflict in calculate_bollinger_bands() - Changed std_dev parameter to num_std_dev to avoid conflict with the calculated standard deviation
Fixed calculation error in Bollinger Bands - Corrected the formula from std_dev * std_dev to num_std_dev * rolling_std
Fixed parentheses error in calculate_bollinger_bands_width() - Corrected the formula to properly calculate (upper_band - lower_band) / middle_band

📊 Test Coverage:
Basic functionality tests - Verify correct return types, lengths, and calculations
Edge case tests - Handle empty data, single values, zero values, insufficient data
Parameter validation - Test invalid parameters and error handling
Integration tests - Test complete workflows using multiple functions together
Data validation - Test behavior with missing columns and invalid inputs

run the tests
python -m pytest test_volatility_metrics.py -v
