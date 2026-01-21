"""
Pytest configuration and shared fixtures for IGA tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tolerance():
    """Default tolerance for floating point comparisons."""
    return 1e-12


@pytest.fixture
def loose_tolerance():
    """Looser tolerance for numerical integration tests."""
    return 1e-8
