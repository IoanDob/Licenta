# app/utils/__init__.py
"""
Utils package initialization with correct imports
"""

# Import from unified processor
from .unified_processor import (
    UnifiedProcessor,
    ColumnType, 
    DetectionResult,
    create_processor
)

# Backward compatibility aliases
SmartDetector = UnifiedProcessor

# Export main components
__all__ = [
    'UnifiedProcessor',
    'SmartDetector',  # Alias for backward compatibility
    'ColumnType',
    'DetectionResult', 
    'create_processor'
]