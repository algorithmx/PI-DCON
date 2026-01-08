"""
Shared training utilities for PI-DCON models.

This module contains common helper functions used across different
physics-informed training tasks (Darcy flow, plate stress, etc.)
"""

import torch
import torch.nn as nn
import torch.optim as optim


def print_training_config(config):
    """Print training configuration details."""
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])
    print('BC weight', config['train']['bc_weight'])
    
    # Print GradNorm configuration if present
    gradnorm_config = config.get('train', {}).get('gradnorm', {})
    if gradnorm_config.get('enabled', False):
        use_simple = gradnorm_config.get('use_simple', False)
        print('GradNorm enabled: True')
        print('  mode:', 'Simple (loss-ratio)' if use_simple else 'Full (gradient-based)')
        print('  alpha:', gradnorm_config.get('alpha', 1.5))
        if use_simple:
            print('  window_size:', gradnorm_config.get('window_size', 50))
        else:
            print('  lr_weights:', gradnorm_config.get('lr_weights', 0.025))
    else:
        print('GradNorm enabled: False')


def setup_optimizer_and_loss(config, model):
    """
    Setup optimizer and loss function.

    Args:
        config: Configuration dictionary containing training parameters
        model: Neural network model

    Returns:
        tuple: (mse_loss, adam_optimizer)
    """
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])
    return mse, optimizer


def try_load_pretrained_model(model, args, device):
    """
    Attempt to load a pre-trained model if it exists.

    Args:
        model: Neural network model
        args: Arguments containing model name and dataset name
        device: Target device (cpu or cuda)

    Note:
        Silently passes if no pre-trained model is found.
    """
    try:
        model.load_state_dict(torch.load(
            r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model),
            map_location=device
        ))
    except:
        print('No pre-trained model found.')


def validate_and_save_model_generic(model, val_loader, coors, device, args, epoch,
                                     min_val_err, avg_losses, vf, val_func):
    """
    Generic validation and model saving function.

    Args:
        model: Neural network model
        val_loader: Validation data loader
        coors: Coordinates for evaluation
        device: Target device
        args: Arguments containing model name and dataset name
        epoch: Current epoch number
        min_val_err: Current minimum validation error
        avg_losses: Dictionary of accumulated losses
        vf: Validation frequency (validate every vf epochs)
        val_func: Validation function to use (val for Darcy, val for plate)

    Returns:
        float: Updated minimum validation error
    """
    if epoch % vf == 0:
        model.eval()

        # Call the appropriate validation function
        if hasattr(val_func, '__code__') and val_func.__code__.co_argcount > 4:
            # For plate validation which returns pointwise_err
            err, pointwise_err = val_func(model, val_loader, coors, device)
        else:
            # For Darcy validation
            err = val_func(model, val_loader, coors, device)
            pointwise_err = None

        print('Best L2 relative error:', err)

        # Print all accumulated losses (unweighted true values)
        for key, value in avg_losses.items():
            if value != 0:  # Only print non-zero losses
                print(f'current period {key} loss (unweighted):', value / vf)

        # Save model if improved
        if err < min_val_err:
            torch.save(model.state_dict(),
                      r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model))
            min_val_err = err

        # Reset recorded losses
        for key in avg_losses:
            avg_losses[key] = 0

        return min_val_err, pointwise_err

    return min_val_err, None
