# This file makes the directory a Python package and exposes the ComfyUI nodes.

# Import the node mappings from your nodes.py file
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Optional: Define required packages for ComfyUI Manager (if you ever make it public)
REQUIRED_PACKAGES = {
    "requests": ">=2.28.1", # The version of requests used in the client_example.py
    "Pillow": ">=9.0.0", # Pillow is usually installed with ComfyUI, but good to list
    "numpy": ">=1.20.0", # Numpy is usually installed with ComfyUI, but good to list
}

# Optional: Define a WEB_DIRECTORY if you have custom web UI elements (JavaScript, CSS)
# WEB_DIRECTORY = "./web" # Uncomment and create a 'web' folder if needed

# Optional: Package metadata
__version__ = "0.0.1"
__author__ = "Matr"