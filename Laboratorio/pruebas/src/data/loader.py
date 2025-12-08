# src/data/loader.py
"""
Data loading module for PQRS dataset
Handles reading and initial validation of PQRS data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial validation of PQRS data.
    
    Attributes:
        required_columns: Columns that must be present in the dataset
        dtype_mapping: Data type specifications for columns
    """
    
    REQUIRED_COLUMNS = [
        "PQRS No.",
        "DESCRIPCION DEL HECHO",
        "ENTIDAD RESPONSABLE",
        "TIPOS DE HECHO",
        "ESTADO",
    ]
    
    DTYPE_MAPPING = {
        "PQRS No.": "int32",
        "DESCRIPCION DEL HECHO": "str",
        "ENTIDAD RESPONSABLE": "str",
        "TIPOS DE HECHO": "str",
        "ESTADO": "str",
    }
    
    def __init__(self):
        """Initialize DataLoader"""
        self.df = None
        self.metadata = {}
    
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load PQRS data from CSV or XLSX file.
        
        Args:
            filepath: Path to data file (CSV or XLSX)
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_data("data/raw/pqrs_2014.csv")
            >>> print(df.shape)
            (150, 37)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        try:
            if suffix == ".csv":
                df = pd.read_csv(filepath, encoding="utf-8")
            elif suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(filepath, engine="openpyxl")
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            logger.info(f"✓ Loaded {len(df)} records from {filepath}")
            self.df = df
            self._compute_metadata()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self) -> Tuple[bool, list]:
        """
        Validate that required columns are present and have valid data.
        
        Returns:
            Tuple of (is_valid: bool, errors: list of error messages)
            
        Example:
            >>> loader = DataLoader()
            >>> loader.load_data("data/raw/pqrs.csv")
            >>> is_valid, errors = loader.validate_data()
            >>> if not is_valid:
            ...     print(f"Validation errors: {errors}")
        """
        if self.df is None:
            return False, ["No data loaded"]
        
        errors = []
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check for excessive missing values
        for col in self.REQUIRED_COLUMNS:
            if col in self.df.columns:
                missing_pct = self.df[col].isna().sum() / len(self.df)
                if missing_pct > 0.3:
                    errors.append(
                        f"Column '{col}' has {missing_pct:.1%} missing values"
                    )
        
        # Check minimum records
        if len(self.df) < 10:
            errors.append(f"Dataset too small: {len(self.df)} records (minimum: 10)")
        
        is_valid = len(errors) == 0
        logger.info(f"Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        
        return is_valid, errors
    
    def _compute_metadata(self):
        """Compute and store metadata about the dataset"""
        self.metadata = {
            "n_records": len(self.df),
            "n_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "missing_values": self.df.isna().sum().to_dict(),
            "data_types": self.df.dtypes.to_dict(),
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
    
    def get_metadata(self) -> dict:
        """
        Get metadata about loaded dataset.
        
        Returns:
            Dictionary with dataset metadata
        """
        return self.metadata.copy()
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for the dataset.
        
        Returns:
            DataFrame with summary statistics
            
        Example:
            >>> summary = loader.get_summary_statistics()
            >>> print(summary)
        """
        if self.df is None:
            return pd.DataFrame()
        
        stats = {
            "Column": [],
            "Non-Null": [],
            "Null": [],
            "Null %": [],
            "Unique Values": [],
            "Data Type": [],
        }
        
        for col in self.df.columns:
            stats["Column"].append(col)
            stats["Non-Null"].append(self.df[col].notna().sum())
            stats["Null"].append(self.df[col].isna().sum())
            stats["Null %"].append(
                f"{self.df[col].isna().sum() / len(self.df) * 100:.1f}%"
            )
            stats["Unique Values"].append(self.df[col].nunique())
            stats["Data Type"].append(str(self.df[col].dtype))
        
        return pd.DataFrame(stats)
