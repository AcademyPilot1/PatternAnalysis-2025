"""
@file    utiles.py
@brief   Environment for model parameters
@author  Aaron Morrow, s4642065
@date    30-10-2025
"""

import os
class ModelParameters():
    """
    Class for storing key parameters and results
    """
    def __init__(self):
        # Hyperparameters
        self.learning_rate   = 0.7e-3     # Initial learning rate
        self.weight_decay    = 0.75e-2    # Weight decay for optimizer
        self.dropout         = 0.0        # Dropout rate for fully connected layers
        self.drop_path_rate  = 0.1        # Probability of dropping an entire network path from an iteration
        self.batch_size      = 128        # Batch size for training and evaluation
        self.depth           = 12         # Number of global filter layers to use in the network
        self.epochs          = 300        # Number of training epochs
        
        # Directoy to data AD_NC folder, see README for folder structure requirements
        # Example personal useage path:
        # base_dir = r"C:\Users\user\..path..\ADNI"
        
        base_dir = r"/home/groups/comp3710/ADNI"
        self.data_file_path = os.path.join(base_dir, "AD_NC")
    
        # Data collection
        self.training_losses            = []
        self.estimated_test_losses      = []
        self.test_losses                = []
        self.estimated_test_accuracy    = []
        self.test_accuracy              = []