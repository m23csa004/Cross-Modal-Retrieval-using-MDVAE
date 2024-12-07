import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_file_integrity(file_path: str, expected_extension: str = None) -> bool:
  
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    if expected_extension and not file_path.lower().endswith(expected_extension.lower()):
        logger.error(f"Invalid file extension for {file_path}. Expected: {expected_extension}")
        return False
    
    if os.path.getsize(file_path) == 0:
        logger.error(f"File is empty: {file_path}")
        return False
        
    return True

def load_examples(split: int = 1, root_path: str = '../wikipedia_dataset/') -> Tuple[List, List, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict, Dict, List, List, List, List]:

    # Verify root path exists and is accessible
    if not os.path.exists(root_path):
        raise ValueError(f"Dataset root path not found: {root_path}")
    
    logger.info(f"Loading dataset from: {os.path.abspath(root_path)}")
    
    # Define and verify required directory structure
    required_paths = {
        'numpy_output': os.path.join(root_path, 'numpy_output'),
        'text_dir': os.path.join(root_path, 'texts'),
        'image_dir': os.path.join(root_path, 'images'),
        'train_mapping': os.path.join(root_path, 'trainset_txt_img_cat.list'),
        'test_mapping': os.path.join(root_path, 'testset_txt_img_cat.list')
    }
    
    # Verify all required directories exist
    for name, path in required_paths.items():
        if not os.path.exists(path):
            raise ValueError(f"Required {name} directory/file not found at: {path}")
    
    # Verify numpy feature files
    numpy_files = {
        'train_text': os.path.join(required_paths['numpy_output'], 'T_tr.npy'),
        'train_image': os.path.join(required_paths['numpy_output'], 'I_tr.npy'),
        'test_text': os.path.join(required_paths['numpy_output'], 'T_te.npy'),
        'test_image': os.path.join(required_paths['numpy_output'], 'I_te.npy')
    }
    
    for name, file_path in numpy_files.items():
        if not verify_file_integrity(file_path, '.npy'):
            raise ValueError(f"Required numpy file {name} is invalid or missing: {file_path}")
    
    # Load pre-extracted features
    try:
        train_text_VGG = np.load(numpy_files['train_text'])
        train_image_VGG = np.load(numpy_files['train_image'])
        test_text_VGG = np.load(numpy_files['test_text'])
        test_image_VGG = np.load(numpy_files['test_image'])
        logger.info("Successfully loaded pre-extracted features")
    except Exception as e:
        raise ValueError(f"Error loading numpy feature files: {str(e)}")

    # Define category mapping
    categories = {
        1: 'art', 2: 'biology', 3: 'geography', 4: 'history',
        5: 'literature', 6: 'media', 7: 'music', 8: 'royalty',
        9: 'sport', 10: 'warfare'
    }

    # Load and verify training mappings
    if not verify_file_integrity(required_paths['train_mapping']):
        raise ValueError("Training mapping file is invalid or empty")
    
    with open(required_paths['train_mapping']) as f:
        train_mappings = [line.strip().split() for line in f.readlines()]

    # Initialize training data structures
    train_text_paths, train_image_paths = [], []
    trainClasses = []
    train_text_classes, train_image_classes = [], []
    train_text_idx_per_class = {}
    train_image_idx_per_class = {}

    # Process training data
    logger.info("Processing training data...")
    for idx, (text_id, img_id, cat_id) in enumerate(train_mappings):
        category = categories[int(cat_id)]
        text_path = os.path.join(required_paths['text_dir'], f'{text_id}.xml')
        image_path = os.path.join(required_paths['image_dir'], category, f'{img_id}.jpg')
        
        # Verify file existence and integrity
        if not verify_file_integrity(text_path, '.xml'):
            logger.warning(f"Training text file invalid or missing: {text_path}")
            continue
            
        if not verify_file_integrity(image_path, '.jpg'):
            logger.warning(f"Training image file invalid or missing: {image_path}")
            continue
        
        # Store paths and update class information
        train_text_paths.append(text_path)
        train_image_paths.append(image_path)

        if category not in trainClasses:
            trainClasses.append(category)
            train_text_idx_per_class[category] = []
            train_image_idx_per_class[category] = []
            
        train_text_idx_per_class[category].append(idx)
        train_image_idx_per_class[category].append(idx)
        train_text_classes.append(category)
        train_image_classes.append(category)

    # Process test data
    logger.info("Processing test data...")
    if not verify_file_integrity(required_paths['test_mapping']):
        raise ValueError("Test mapping file is invalid or empty")
        
    with open(required_paths['test_mapping']) as f:
        test_mappings = [line.strip().split() for line in f.readlines()]

    # Initialize test data structures
    test_text_paths, test_image_paths = [], []
    testClasses = []
    test_text_classes, test_image_classes = [], []
    test_text_idx_per_class = {}
    test_image_idx_per_class = {}
    tsc, tic = [], []  # Compatibility lists

    # Process test data
    for idx, (text_id, img_id, cat_id) in enumerate(test_mappings):
        category = categories[int(cat_id)]
        text_path = os.path.join(required_paths['text_dir'], f'{text_id}.xml')
        image_path = os.path.join(required_paths['image_dir'], category, f'{img_id}.jpg')
        
        # Verify file existence and integrity
        if not verify_file_integrity(text_path, '.xml'):
            logger.warning(f"Test text file invalid or missing: {text_path}")
            continue
            
        if not verify_file_integrity(image_path, '.jpg'):
            logger.warning(f"Test image file invalid or missing: {image_path}")
            continue
        
        # Store paths and update class information
        test_text_paths.append(text_path)
        test_image_paths.append(image_path)
        
        if category not in testClasses:
            testClasses.append(category)
            test_text_idx_per_class[category] = []
            test_image_idx_per_class[category] = []
            
        test_text_idx_per_class[category].append(idx)
        test_image_idx_per_class[category].append(idx)
        test_text_classes.append(category)
        test_image_classes.append(category)
        tsc.append(category)
        tic.append(category)

    logger.info(f"Loaded {len(train_text_paths)} training samples and {len(test_text_paths)} test samples")
    
    return (trainClasses, testClasses, 
            train_text_VGG, train_image_VGG, test_text_VGG, test_image_VGG,
            train_text_idx_per_class, train_image_idx_per_class, test_text_idx_per_class,
            train_text_classes,train_image_classes,test_text_classes, test_image_classes, tsc, tic)

