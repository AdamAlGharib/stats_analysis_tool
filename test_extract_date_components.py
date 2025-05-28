import unittest
from datetime import datetime
from daily_returns import extract_date_components


class TestExtractDateComponents(unittest.TestCase):
    """Test cases for the extract_date_components function."""
    
    def test_string_timestamp_basic(self):
        """Test with a basic string timestamp."""
        result = extract_date_components("2023-10-26 14:30:00")
        expected = {
            "year": 2023,
            "quarter": 4,
            "month_number": 10,
            "month_name": "October",
            "day_of_week_number": 3,  # Thursday (0=Mon, 3=Thu)
            "day_of_week_name": "Thursday"
        }
        self.assertEqual(result, expected)
    
    def test_datetime_object_basic(self):
        """Test with a datetime object."""
        dt = datetime(2023, 10, 26, 14, 30, 0)
        result = extract_date_components(dt)
        expected = {
            "year": 2023,
            "quarter": 4,
            "month_number": 10,
            "month_name": "October",
            "day_of_week_number": 3,  # Thursday
            "day_of_week_name": "Thursday"
        }
        self.assertEqual(result, expected)
    
    def test_quarter_boundaries(self):
        """Test quarter calculation at boundaries."""
        # Q1
        result_q1 = extract_date_components("2023-01-01 00:00:00")
        self.assertEqual(result_q1["quarter"], 1)
        
        result_q1_end = extract_date_components("2023-03-31 23:59:59")
        self.assertEqual(result_q1_end["quarter"], 1)
        
        # Q2
        result_q2 = extract_date_components("2023-04-01 00:00:00")
        self.assertEqual(result_q2["quarter"], 2)
        
        # Q3
        result_q3 = extract_date_components("2023-07-01 00:00:00")
        self.assertEqual(result_q3["quarter"], 3)
        
        # Q4
        result_q4 = extract_date_components("2023-12-31 23:59:59")
        self.assertEqual(result_q4["quarter"], 4)
    
    def test_different_days_of_week(self):
        """Test all days of the week."""
        test_cases = [
            ("2023-10-23 12:00:00", "Monday", 0),
            ("2023-10-24 12:00:00", "Tuesday", 1),
            ("2023-10-25 12:00:00", "Wednesday", 2),
            ("2023-10-26 12:00:00", "Thursday", 3),
            ("2023-10-27 12:00:00", "Friday", 4),
            ("2023-10-28 12:00:00", "Saturday", 5),
            ("2023-10-29 12:00:00", "Sunday", 6),
        ]
        
        for date_str, expected_day_name, expected_day_num in test_cases:
            result = extract_date_components(date_str)
            self.assertEqual(result["day_of_week_name"], expected_day_name)
            self.assertEqual(result["day_of_week_number"], expected_day_num)
    
    def test_different_months(self):
        """Test all months of the year."""
        months = [
            (1, "January"), (2, "February"), (3, "March"), (4, "April"),
            (5, "May"), (6, "June"), (7, "July"), (8, "August"),
            (9, "September"), (10, "October"), (11, "November"), (12, "December")
        ]
        
        for month_num, month_name in months:
            date_str = f"2023-{month_num:02d}-15 12:00:00"
            result = extract_date_components(date_str)
            self.assertEqual(result["month_number"], month_num)
            self.assertEqual(result["month_name"], month_name)
    
    def test_leap_year(self):
        """Test with leap year date."""
        result = extract_date_components("2024-02-29 12:00:00")
        self.assertEqual(result["year"], 2024)
        self.assertEqual(result["month_number"], 2)
        self.assertEqual(result["day_of_week_name"], "Thursday")
    
    def test_edge_cases_time(self):
        """Test edge cases with different times."""
        # Midnight
        result_midnight = extract_date_components("2023-01-01 00:00:00")
        self.assertEqual(result_midnight["year"], 2023)
        
        # Just before midnight
        result_late = extract_date_components("2023-12-31 23:59:59")
        self.assertEqual(result_late["year"], 2023)
        self.assertEqual(result_late["month_number"], 12)
    
    def test_invalid_string_format(self):
        """Test with invalid string formats."""
        invalid_formats = [
            "2023-10-26",  # Missing time
            "10/26/2023 14:30:00",  # Wrong date format
            "2023-10-26T14:30:00",  # ISO format with T
            "26-10-2023 14:30:00",  # DD-MM-YYYY format
            "2023/10/26 14:30:00",  # Slashes instead of dashes
            "invalid date string",  # Completely invalid
        ]
        
        for invalid_str in invalid_formats:
            with self.assertRaises(ValueError) as context:
                extract_date_components(invalid_str)
            self.assertIn("Timestamp string must match", str(context.exception))
    
    def test_invalid_input_type(self):
        """Test with invalid input types."""
        invalid_inputs = [
            123456789,  # Integer
            12.34,  # Float
            ["2023-10-26 14:30:00"],  # List
            {"date": "2023-10-26 14:30:00"},  # Dict
            None,  # None
        ]
        
        for invalid_input in invalid_inputs:
            with self.assertRaises(TypeError) as context:
                extract_date_components(invalid_input)
            self.assertIn("must be a datetime object or a timestamp string", 
                         str(context.exception))
    
    def test_year_boundaries(self):
        """Test with different years including century boundaries."""
        test_years = [
            ("1999-12-31 23:59:59", 1999),
            ("2000-01-01 00:00:00", 2000),
            ("2099-12-31 23:59:59", 2099),
            ("2100-01-01 00:00:00", 2100),
        ]
        
        for date_str, expected_year in test_years:
            result = extract_date_components(date_str)
            self.assertEqual(result["year"], expected_year)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2) 