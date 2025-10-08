"""
Data Loader Module

This module handles loading, validation, and preprocessing of raw Nifty 1-minute data.
It ensures data quality and prepares it for feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import logging

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and validation of Nifty OHLCV data.
    
    This class ensures that the input data is properly formatted, contains all required
    columns, has valid datetime indices, and is free from obvious data quality issues.
    
    Attributes:
        file_path (Path): Path to the raw data CSV file
        df (pd.DataFrame): Loaded and validated dataframe
    """
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the DataLoader with a file path.
        
        Parameters:
            file_path (str or Path): Path to the CSV file containing OHLCV data
        """
        self.file_path = Path(file_path)
        self.df = None
        
        # Validate that the file exists before proceeding
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        logger.info(f"DataLoader initialized with file: {self.file_path}")
    
    def load_and_validate(self) -> pd.DataFrame:
        """
        Load CSV data and perform comprehensive validation.
        
        This method orchestrates the entire data loading pipeline:
        1. Read the CSV file
        2. Parse and validate datetime columns
        3. Validate OHLCV columns exist and have correct data types
        4. Check for data quality issues (duplicates, nulls, invalid values)
        5. Sort by datetime
        
        Returns:
            pd.DataFrame: Clean, validated dataframe with datetime index
            
        Raises:
            ValueError: If data validation fails
        """
        logger.info("Loading data from CSV...")
        
        # Step 1: Read the CSV file
        # We use low_memory=False to ensure consistent data type inference
        self.df = pd.read_csv(self.file_path, low_memory=False)
        logger.info(f"Loaded {len(self.df)} rows from {self.file_path.name}")
        
        # Step 2: Parse datetime columns and set as index
        self._parse_datetime()
        
        # Step 3: Validate that all required OHLCV columns are present
        self._validate_columns()
        
        # Step 4: Validate data types and convert if necessary
        self._validate_data_types()
        
        # Step 5: Check for and handle data quality issues
        self._check_data_quality()
        
        # Step 6: Sort by datetime to ensure temporal order
        # This is crucial for time-series analysis
        self.df.sort_index(inplace=True)
        logger.info("Data sorted by datetime")
        
        # Step 7: Remove any duplicate timestamps
        # Keep the first occurrence if duplicates exist
        duplicates = self.df.index.duplicated(keep='first')
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate timestamps. Keeping first occurrence.")
            self.df = self.df[~duplicates]
        
        logger.info(f"Data validation complete. Final shape: {self.df.shape}")
        return self.df
    
    def _parse_datetime(self) -> None:
        """
        Parse datetime columns and set as dataframe index.
        
        This method handles two common datetime formats:
        1. Separate 'Date' and 'Time' columns
        2. Single 'Datetime' column
        
        The datetime index is essential for time-series operations and ensures
        that all subsequent operations maintain temporal order.
        """
        try:
            # Check if we have separate date and time columns
            if all(col in self.df.columns for col in config.DATE_COLUMNS):
                logger.info("Parsing separate date and time columns...")
                
                # Combine date and time into a single datetime column
                # Format: 'YYYY-MM-DD HH:MM:SS'
                self.df['datetime'] = pd.to_datetime(
                    self.df[config.DATE_COLUMNS[0]] + ' ' + self.df[config.DATE_COLUMNS[1]],
                    format='mixed',  # Automatically infer the format
                    errors='coerce'  # Convert parsing errors to NaT (Not a Time)
                )
                
                # Drop the original date and time columns as we now have datetime
                self.df.drop(columns=config.DATE_COLUMNS, inplace=True)
            
            # Check if we have a single datetime column
            elif 'Datetime' in self.df.columns or 'datetime' in self.df.columns:
                datetime_col = 'Datetime' if 'Datetime' in self.df.columns else 'datetime'
                logger.info(f"Parsing datetime column: {datetime_col}")
                
                # Convert to datetime, handling various formats automatically
                self.df['datetime'] = pd.to_datetime(
                    self.df[datetime_col],
                    format='mixed',
                    errors='coerce'
                )
                
                # Drop the original datetime column if it had a different name
                if datetime_col != 'datetime':
                    self.df.drop(columns=[datetime_col], inplace=True)
            
            else:
                raise ValueError(
                    "No datetime columns found. Expected either "
                    f"{config.DATE_COLUMNS} or 'Datetime' column"
                )
            
            # Check for any timestamps that failed to parse (NaT values)
            nat_count = self.df['datetime'].isna().sum()
            if nat_count > 0:
                logger.warning(f"Found {nat_count} unparseable datetime values. These rows will be removed.")
                self.df.dropna(subset=['datetime'], inplace=True)
            
            # Set datetime as the index for time-series operations
            self.df.set_index('datetime', inplace=True)
            logger.info("Datetime index set successfully")
            
        except Exception as e:
            raise ValueError(f"Error parsing datetime columns: {str(e)}")
    
    def _validate_columns(self) -> None:
        """
        Validate that all required OHLCV columns are present in the dataframe.
        
        Raises:
            ValueError: If any required column is missing
        """
        required_cols = list(config.OHLCV_COLUMNS.values())
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        logger.info(f"All required OHLCV columns present: {required_cols}")
    
    def _validate_data_types(self) -> None:
        """
        Validate and convert OHLCV columns to appropriate numeric types.
        
        All OHLCV columns should be numeric (float or int). This method ensures
        proper data types and converts strings to numbers where possible.
        """
        ohlcv_cols = list(config.OHLCV_COLUMNS.values())
        
        for col in ohlcv_cols:
            # Check if column is already numeric
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                logger.warning(f"Column {col} is not numeric. Attempting to convert...")
                
                try:
                    # Convert to numeric, coercing errors to NaN
                    # This handles cases where data has commas, spaces, or other formatting
                    self.df[col] = pd.to_numeric(
                        self.df[col].astype(str).str.replace(',', ''),
                        errors='coerce'
                    )
                    logger.info(f"Successfully converted {col} to numeric")
                except Exception as e:
                    raise ValueError(f"Could not convert {col} to numeric: {str(e)}")
        
        # Convert volume to integer type (volumes are counts, not decimals)
        volume_col = config.OHLCV_COLUMNS['volume']
        self.df[volume_col] = self.df[volume_col].fillna(0).astype('int64')
        
        logger.info("Data type validation complete")
    
    def _check_data_quality(self) -> None:
        """
        Perform comprehensive data quality checks and handle issues.
        
        This method checks for:
        1. Null values in OHLCV columns
        2. Invalid price relationships (high < low, close outside high-low range)
        3. Negative or zero prices
        4. Negative volumes
        5. Extreme outliers (optional)
        """
        ohlcv_cols = list(config.OHLCV_COLUMNS.values())
        
        # Check 1: Null values
        null_counts = self.df[ohlcv_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
            
            # For now, we'll drop rows with nulls in OHLCV columns
            # Alternative: use forward fill or interpolation
            initial_len = len(self.df)
            self.df.dropna(subset=ohlcv_cols, inplace=True)
            logger.info(f"Removed {initial_len - len(self.df)} rows with null values")
        
        # Check 2: Invalid price relationships
        # High should be >= Low
        invalid_hl = self.df[config.OHLCV_COLUMNS['high']] < self.df[config.OHLCV_COLUMNS['low']]
        if invalid_hl.any():
            logger.error(f"Found {invalid_hl.sum()} rows where High < Low")
            # Remove these invalid rows
            self.df = self.df[~invalid_hl]
        
        # Close should be between Low and High
        close_col = config.OHLCV_COLUMNS['close']
        high_col = config.OHLCV_COLUMNS['high']
        low_col = config.OHLCV_COLUMNS['low']
        
        invalid_close = (
            (self.df[close_col] > self.df[high_col]) | 
            (self.df[close_col] < self.df[low_col])
        )
        if invalid_close.any():
            logger.error(f"Found {invalid_close.sum()} rows where Close is outside High-Low range")
            self.df = self.df[~invalid_close]
        
        # Check 3: Negative or zero prices
        # Prices should always be positive
        for col in ['open', 'high', 'low', 'close']:
            col_name = config.OHLCV_COLUMNS[col]
            invalid_price = self.df[col_name] <= 0
            if invalid_price.any():
                logger.error(f"Found {invalid_price.sum()} rows with non-positive {col}")
                self.df = self.df[~invalid_price]
        
        # Check 4: Negative volumes
        volume_col = config.OHLCV_COLUMNS['volume']
        invalid_volume = self.df[volume_col] < 0
        if invalid_volume.any():
            logger.error(f"Found {invalid_volume.sum()} rows with negative volume")
            self.df = self.df[~invalid_volume]
        
        logger.info("Data quality checks complete")
    
    def get_date_range(self) -> tuple:
        """
        Get the date range of the loaded data.
        
        Returns:
            tuple: (start_date, end_date) as pandas Timestamp objects
        """
        if self.df is None or self.df.empty:
            return None, None
        
        return self.df.index.min(), self.df.index.max()
    
    def get_trading_days_count(self) -> int:
        """
        Count the number of unique trading days in the dataset.
        
        Returns:
            int: Number of unique dates
        """
        if self.df is None or self.df.empty:
            return 0
        
        return self.df.index.date.nunique()
    
    def save_processed_data(self, output_path: Optional[Path] = None) -> None:
        """
        Save the validated dataframe to a CSV file.
        
        Parameters:
            output_path (Path, optional): Where to save the file. 
                                         Defaults to config.PROCESSED_DATA_DIR
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data to save. Load and validate data first.")
        
        if output_path is None:
            output_path = config.PROCESSED_DATA_DIR / 'validated_data.csv'
        
        self.df.to_csv(output_path)
        logger.info(f"Processed data saved to {output_path}")
