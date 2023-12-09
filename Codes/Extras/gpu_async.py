# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 05:37:13 2023

@author: XENo
"""

import os

# Set the environment variable
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Now, run your TensorFlow code
runfile('C:/Users/XENo/Desktop/Thesis/Thesis_try - dropout.py', wdir='C:/Users/XENo/Desktop/Thesis')
