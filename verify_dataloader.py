import numpy as np
from dataloader import load_examples
import os
from typing import Any, Dict, List, Tuple

def verify_dataloader(root_path: str) -> None:
   
    print("\n=== Dataset Loader Verification ===\n")
    
    try:
        # Load the dataset
        print("Loading dataset...")
        result = load_examples(root_path=root_path)
        
        # Unpack all returned values
        (trainClasses, testClasses, 
         train_text_VGG, train_image_VGG, 
         test_text_VGG, test_image_VGG,
         train_text_idx_per_class, train_image_idx_per_class, 
         test_text_idx_per_class,
         train_text_classes,train_image_classes,
         test_text_classes, test_image_classes, 
         tsc, tic) = result
        
        print("\n1. Basic Data Checks:")
        print("-" * 50)
        
        # Check classes
        print(f"\nUnique Training Classes ({len(trainClasses)}):", trainClasses)
        print(f"Unique Test Classes ({len(testClasses)}):", testClasses)
        
        # Check feature arrays
        print("\n2. Feature Array Shapes:")
        print("-" * 50)
        print(f"Train Text Features: {train_text_VGG.shape}")
        print(f"Train Image Features: {train_image_VGG.shape}")
        print(f"Test Text Features: {test_text_VGG.shape}")
        print(f"Test Image Features: {test_image_VGG.shape}")
        
        # Check class distributions
        print("\n3. Class Distribution:")
        print("-" * 50)
        from collections import Counter
        print("\nTraining Class Distribution:")
        train_dist = Counter(train_text_classes)
        for cls, count in train_dist.items():
            print(f"{cls}: {count} samples")
            
        print("\nTest Class Distribution:")
        test_dist = Counter(test_image_classes)
        for cls, count in test_dist.items():
            print(f"{cls}: {count} samples")
        
        # Verify indices per class
        print("\n4. Indices per Class Check:")
        print("-" * 50)
        print("\nTraining Indices Summary:")
        for cls in trainClasses:
            text_indices = train_text_idx_per_class.get(cls, [])
            image_indices = train_image_idx_per_class.get(cls, [])
            print(f"{cls}: {len(text_indices)} text indices, {len(image_indices)} image indices")
        
        # Check for data integrity
        print("\n5. Data Integrity Checks:")
        print("-" * 50)
        
        # Check for NaN values
        print("\nChecking for NaN values:")
        print(f"Train Text NaN: {np.isnan(train_text_VGG).any()}")
        print(f"Train Image NaN: {np.isnan(train_image_VGG).any()}")
        print(f"Test Text NaN: {np.isnan(test_text_VGG).any()}")
        print(f"Test Image NaN: {np.isnan(test_image_VGG).any()}")
        
        # Check value ranges
        print("\nFeature value ranges:")
        print(f"Train Text: [{train_text_VGG.min():.3f}, {train_text_VGG.max():.3f}]")
        print(f"Train Image: [{train_image_VGG.min():.3f}, {train_image_VGG.max():.3f}]")
        print(f"Test Text: [{test_text_VGG.min():.3f}, {test_text_VGG.max():.3f}]")
        print(f"Test Image: [{test_image_VGG.min():.3f}, {test_image_VGG.max():.3f}]")
        
        # Verify matching lengths
        print("\n6. Length Consistency Checks:")
        print("-" * 50)
        checks = [
            ("Train text and image features", train_text_VGG.shape[0] == train_image_VGG.shape[0]),
            ("Test text and image features", test_text_VGG.shape[0] == test_image_VGG.shape[0]),
            ("Test text classes and features", len(test_text_classes) == test_text_VGG.shape[0]),
            ("Test image classes and features", len(test_image_classes) == test_image_VGG.shape[0]),
            ("TSC and TIC lengths", len(tsc) == len(tic))
        ]
        
        for check_name, result in checks:
            print(f"{check_name}: {'✓ PASS' if result else '✗ FAIL'}")
        
        print("\n=== Verification Complete ===")
        
    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}")
        raise

if __name__ == "__main__":
    # Replace with your dataset path
    dataset_path = '../wikipedia_dataset/'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist!")
    else:
        verify_dataloader(dataset_path)