class MercuryError(Exception):
    """Base exception for Mercury package."""
    pass

class DataLoadError(MercuryError):
    """Raised when data loading fails."""
    pass

class ConfigError(MercuryError):
    """Raised for configuration issues."""
    pass 