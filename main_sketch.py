from dataloader import load_examples
from models import CMR_NET
from config import config_sketchy as config
from utils import mapChange, random_train_X, sample_normal
import numpy as np
import torch
import random
from sklearn.neighbors import NearestNeighbors
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(epochs, model):
    # Initialize global step
    GLOBAL_step = 0
    llist = []
    
    # Load data
    try:
        (trainClasses, testClasses, 
         train_text_VGG, train_image_VGG, 
         test_text_VGG, test_image_VGG,
         train_text_idx_per_class, train_image_idx_per_class, 
         test_text_idx_per_class,
         train_text_classes, train_image_classes,
         test_text_classes, test_image_classes,
         tsc, tic) = load_examples()
        
        logger.info(f"Successfully loaded dataset with {len(trainClasses)} training classes")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

    # Calculate steps per epoch based on batch size
    train_steps_per_epoch = len(trainClasses) // config['batch']
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        model = model.cuda()
        
        for step in range(train_steps_per_epoch):
            GLOBAL_step += 1
            
            # Get random training samples using utility function
            train_X_text_idx, train_X_image_idx = random_train_X(
                trainClasses,
                train_text_idx_per_class,  # Using text indices instead of sketch
                train_image_idx_per_class
            )
            
            # Prepare batch data
            X_text = np.take(train_text_VGG, train_X_text_idx, axis=0)
            X_image = np.take(train_image_VGG, train_X_image_idx, axis=0)
            X_LABEL = np.eye(config['class_num'])
            X_LABEL = np.argmax(X_LABEL, axis=1)

            # Move to GPU
            X = torch.Tensor(X_text).cuda()
            Y = torch.Tensor(X_image).cuda()
            LABEL = torch.Tensor(X_LABEL).to(torch.int64).detach().cuda()
            
            # Forward pass and loss calculation
            # Note: train_func calculation moved to model for clarity
            outputs = model(X, Y, LABEL)
            
            # Sample from normal distribution if needed
            if hasattr(outputs, 'logvar') and hasattr(outputs, 'mean'):
                sampled = sample_normal(outputs.logvar, outputs.mean, use_cuda=True, istraining=True)
                outputs.update({'sampled': sampled})
            
            # Calculate loss using model's loss function
            loss_dict = model.calculate_loss(outputs, GLOBAL_step)
            
            # Optimization step
            optimizer.zero_grad()
            loss_dict['joint_loss'].backward()
            optimizer.step()
            
            llist.append(loss_dict['joint_loss'].detach().cpu().numpy().tolist())

    # Evaluation phase
    logger.info("Starting evaluation phase")
    with torch.no_grad():
        model.eval()
        model = model.cpu()
        X_SharedFeat = []
        Y_SharedFeat = []
        
        # Process test text features
        total_num = len(test_text_classes)
        test_max_steps = (total_num + config['batch'] - 1) // config['batch']
        
        # Extract text features
        for test_step in range(test_max_steps):
            start_idx = test_step * config['batch']
            end_idx = min((test_step + 1) * config['batch'], total_num)
            X = torch.Tensor(test_text_VGG[start_idx:end_idx])
            
            if X.shape[0] < config['batch']:
                padding = config['batch'] - X.shape[0]
                X = torch.cat([X, torch.zeros((padding, X.shape[1]))], dim=0)
            
            results = model.text_encoder(X)
            X_SharedFeat.append(results.numpy())

        X_SharedFeat = np.concatenate(X_SharedFeat, axis=0)[:total_num]

        # Extract image features
        total_num = len(test_image_classes)
        test_max_steps = (total_num + config['batch'] - 1) // config['batch']
        
        for test_step in range(test_max_steps):
            start_idx = test_step * config['batch']
            end_idx = min((test_step + 1) * config['batch'], total_num)
            Y = torch.Tensor(test_image_VGG[start_idx:end_idx])
            
            if Y.shape[0] < config['batch']:
                padding = config['batch'] - Y.shape[0]
                Y = torch.cat([Y, torch.zeros((padding, Y.shape[1]))], dim=0)
            
            results = model.image_encoder(Y)
            Y_SharedFeat.append(results.numpy())

        Y_SharedFeat = np.concatenate(Y_SharedFeat, axis=0)[:total_num]

        # Evaluation using K-Nearest Neighbors
        K = 5  # Can be moved to config if needed
        nbrs = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute').fit(Y_SharedFeat)
        distances, indices = nbrs.kneighbors(X_SharedFeat)
        
        # Calculate precision@K
        retrieved_classes = np.array(test_image_classes)[indices]
        results = np.zeros(retrieved_classes.shape)
        for idx in range(results.shape[0]):
            results[idx] = (retrieved_classes[idx] == np.array(test_text_classes)[idx])
        precision_K = np.mean(results, axis=1)
        
        logger.info(f'Mean precision@{K} for test texts: {np.mean(precision_K)}')

        # Calculate mAP using utility function
        nbrs_all = NearestNeighbors(
            n_neighbors=Y_SharedFeat.shape[0],
            metric='cosine',
            algorithm='brute'
        ).fit(Y_SharedFeat)
        
        distances, indices = nbrs_all.kneighbors(X_SharedFeat)
        retrieved_classes = np.array(test_image_classes)[indices]
        results = np.zeros(retrieved_classes.shape)
        gt_count = []
        
        for idx in range(results.shape[0]):
            results[idx] = (retrieved_classes[idx] == np.array(test_text_classes)[idx])
            gt_count.append(np.sum(results[idx]))
            
        gt_count = np.array(gt_count)
        temp = [np.arange(results.shape[1]) for _ in range(results.shape[0])]
        mAP_term = 1.0 / (np.stack(temp, axis=0) + 1)
        mAP = np.sum(np.multiply(mapChange(results), mAP_term), axis=1)
        mAP = mAP / gt_count
        
        logger.info(f'Mean mAP@all for test texts: {np.mean(mAP)}')
        
        return model, np.mean(mAP), llist

if __name__ == "__main__":
    logger.info(f"Configuration: {config}")
    
    try:
        # Initialize model with config parameters
        model = CMR_NET(
            config=config,
            use_cuda=True
        )
        
        # Initialize optimizer with learning rate from config if available
        # Add this check before creating the optimizer
        if not list(model.parameters()):
            raise ValueError("Model has no parameters!")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.002))
        #optimizer = torch.optim.Adam(model.parameters(),lr=config.get('learning_rate', 0.002))
        
        # Run training
        model, mmap, loss_list = main(10, model)
        logger.info(f"Training completed successfully with final mAP: {mmap}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
