
import os
class ModelParameters():
    """
    Class for storing key parameters
    """
    def __init__(self):
        self.learning_rate  = 0.7e-3
        self.weight_decay   = 0.75e-2
        self.dropout        = 0.0
        self.drop_path_rate      = 0.1       # Probability of dropping an entire network path from an iteration
        self.batch_size     = 128

        self.depth          = 12         # Number of global filter layers to use in the network
        self.epochs         = 200
        
        
        base_dir = r"C:\Users\acade\Documents\UNI\Sem 2 2025\COMP3710\ADNI"
        self.data_file_path = os.path.join(base_dir, "AD_NC")
    
        # C:/Users/acade/Documents/UNI/Sem 2 2025/COMP3710/ADNI/AD_NC
        # C:/Users/acade/Documents/UNI/Sem 2 2025/COMP3710/ADNI/AD_NC
        
        # C:/Users/acade/Documents/UNI/Sem 2 2025/COMP3710/PatternAnalysis-2025/recognition/ConvNeXt_s4642065/ConvNeXt_best.pth
        # Data collection
        self.training_losses            = []
        self.estimated_test_losses      = []
        self.test_losses                = []
        self.estimated_test_accuracy    = []
        self.test_accuracy              = []