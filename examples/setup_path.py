"""
Path setup for examples

This module sets up the Python path to allow importing apilot package
without installing it. Import this at the beginning of all example scripts.

Usage:
    import setup_path
    from apilot.core import ...  # Now you can import apilot modules
"""

import os
import sys


def add_project_to_path():
    """Add the project root directory to Python path."""
    # Get the directory containing this file (examples)
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root directory (parent of examples)
    project_root = os.path.dirname(examples_dir)
    
    # Add project root to path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# Automatically add project to path when this module is imported
add_project_to_path()
