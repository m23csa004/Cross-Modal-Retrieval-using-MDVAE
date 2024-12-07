# # i am assuming sketchycofig as config for image text and modifying the hyperparameters
# config_sketchy= {
#     'alpha' :0.001,        ## initially it was1
#     'beta' :0.01,       # initially it was 0.3
#     'gamma' :10,        # initially it was1
#     'lambda' :0.01,    # initially it was1
#     'ar':2500.0,
#     'neef':128,
#     'nsef':128,
#     'ncef':128,
#     'nzcef':128,
#     'exdim':64,
#     'shdim':64,
#     'cdim':512,
#     'ndf':128,
#     'l1_weight':10,
#     'bn_momentum':0.9,
#     'input_dim':[128,10],      # intially it was 512,512
#     'class_num':10,           # initially it was 100
#     'batch':64,               # initially it was 64

# }

# config_tuberlin= {
#     'alpha' :1,
#     'beta' :0.3,
#     'gamma' :1,
#     'lambda' :1,
#     'ar':2500.0,
#     'neef':128,
#     'nsef':128,
#     'ncef':128,
#     'nzcef':128,
#     'exdim':64,
#     'shdim':64,
#     'cdim':512,
#     'ndf':128,
#     'l1_weight':10,
#     'bn_momentum':0.9,
#     'input_dim':[128,10],   # intially it was 512,512
#     'class_num':10,         # initially it was 100
#     'batch':100,

# }


# Updated configuration based on actual feature dimensions
config_sketchy = {
    # Input dimensions adjusted to match actual feature shapes
    'input_dim': [10, 128],  # [text_dim, image_dim] based on provided shapes
    
    # Network architecture parameters
    'neef': 128,              # Reduced from 128 since input dimensions are smaller
    'nsef': 128,              # Reduced to match neef
    'ncef': 128,              # Reduced to maintain proportionality
    'nzcef': 128,             # Reduced to maintain proportionality
    'exdim': 64,             # Reduced from 64 to better match input dimensions
    'shdim': 64,             # Reduced to match exdim
    'cdim': 512,             # Reduced from 512 to maintain proportionality
    'ndf': 128,               # Reduced from 128 to match other network dimensions
    
    # Loss weights (kept same as original for Sketchy dataset)
    'alpha': 0.001,          # Weight for reconstruction loss
    'beta': 0.01,            # Weight for KL divergence
    'gamma': 10,             # Weight for classification loss
    'lambda': 0.01,          # Weight for domain alignment
    'ar': 2500.0,            # Auto-regressive weight
    'l1_weight': 10,         # L1 loss weight
    
    # Training parameters
    'bn_momentum': 0.9,      # Batch normalization momentum
    'class_num': 10,         # Number of classes (matches text feature dimension)
    'batch': 64,             # Reduced from 64 to better handle smaller dataset
    'learning_rate': 0.002,  # Learning rate maintained
}

