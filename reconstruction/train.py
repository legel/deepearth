#!/usr/bin/env python3
"""
Simple training launcher script
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main function from the multimodal script
from multimodal_plant_training import main

if __name__ == "__main__":
    print("Starting multimodal plant training...")
    print("-" * 60)
    main()
