"""
Inserts the neccessary include directories for python to import project files.
Written by Songrun
"""

import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(root, "PerVertex"))