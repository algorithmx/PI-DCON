import matplotlib.pyplot as plt
import numpy as np
import torch


def prepare_test_coordinates(coors, device):
    """
    Split coordinates into x and y components and move to device.

    Args:
        coors: Array of shape (M, 2) containing coordinates
        device: torch device

    Returns:
        tuple: (test_coor_x, test_coor_y) as unsqueezed tensors on device
    """
    test_coor_x = coors[:, 0].unsqueeze(0).float().to(device)
    test_coor_y = coors[:, 1].unsqueeze(0).float().to(device)
    return test_coor_x, test_coor_y


def initialize_error_tracking():
    """
    Initialize variables for tracking errors during testing.

    Returns:
        dict: Dictionary with initialized tracking variables
    """
    return {
        'mean_relative_L2': 0,
        'num': 0,
        'max_relative_err': -1,
        'min_relative_err': np.inf
    }


def update_best_worst_cases(L2_relative, pred, out, tracking, num_outputs=1):
    """
    Update best and worst error cases during testing.

    Args:
        L2_relative: Tensor of relative L2 errors for current batch
        pred: Model predictions (B, M) or tuple of (u_pred, v_pred)
        out: Ground truth values (B, M) or tuple of (u, v)
        tracking: Dictionary containing current tracking state
        num_outputs: Number of outputs (1 for single output, 2 for multi-output)

    Returns:
        dict: Updated tracking dictionary
    """
    max_err, max_err_idx = torch.topk(L2_relative, 1)
    if max_err > tracking['max_relative_err']:
        tracking['max_relative_err'] = max_err

        if num_outputs == 1:
            tracking['worst_f'] = pred[max_err_idx, :].detach().cpu().numpy()
            tracking['worst_gt'] = out[max_err_idx, :].detach().cpu().numpy()
        else:
            u_pred, v_pred = pred
            u, v = out
            tracking['worst_f'] = u_pred[max_err_idx, :].detach().cpu().numpy()
            tracking['worst_gt'] = u[max_err_idx, :].detach().cpu().numpy()

    min_err, min_err_idx = torch.topk(-L2_relative, 1)
    min_err = -min_err
    if min_err < tracking['min_relative_err']:
        tracking['min_relative_err'] = min_err

        if num_outputs == 1:
            tracking['best_f'] = pred[min_err_idx, :].detach().cpu().numpy()
            tracking['best_gt'] = out[min_err_idx, :].detach().cpu().numpy()
        else:
            u_pred, v_pred = pred
            u, v = out
            tracking['best_f'] = u_pred[min_err_idx, :].detach().cpu().numpy()
            tracking['best_gt'] = u[min_err_idx, :].detach().cpu().numpy()

    return tracking


def accumulate_error(L2_relative, tracking):
    """
    Accumulate error statistics during testing.

    Args:
        L2_relative: Tensor of relative L2 errors for current batch
        tracking: Dictionary containing current tracking state

    Returns:
        dict: Updated tracking dictionary
    """
    tracking['mean_relative_L2'] += torch.sum(L2_relative).detach().cpu().item()
    return tracking


def finalize_error_statistics(tracking, total_samples):
    """
    Finalize error statistics after testing loop.

    Args:
        tracking: Dictionary containing accumulated tracking state
        total_samples: Total number of samples processed

    Returns:
        float: Mean relative L2 error
    """
    return tracking['mean_relative_L2'] / total_samples


def compute_color_range(tracking):
    """
    Compute appropriate color bar range for visualization.

    Args:
        tracking: Dictionary containing best and worst cases

    Returns:
        tuple: (min_color, max_color)
    """
    max_color = np.amax([np.amax(tracking['worst_gt']), np.amax(tracking['best_gt'])])
    min_color = np.amin([np.amin(tracking['worst_gt']), np.amin(tracking['best_gt'])])
    return min_color, max_color


def create_test_visualization(coor_x, coor_y, tracking, scatter_size=5):
    """
    Create visualization of test results with 6 subplots.

    Args:
        coor_x: x-coordinates as numpy array
        coor_y: y-coordinates as numpy array
        tracking: Dictionary containing best and worst cases with 'worst_f', 'worst_gt',
                  'best_f', 'best_gt' as numpy arrays
        scatter_size: Size of scatter plot markers (default: 5)

    Returns:
        None (displays and saves figure)
    """
    min_color, max_color = compute_color_range(tracking)
    cm = plt.cm.get_cmap('RdYlBu')

    plt.figure(figsize=(15, 8), dpi=400)

    # Worst case row
    plt.subplot(2, 3, 1)
    plt.scatter(coor_x, coor_y, c=tracking['worst_f'], cmap=cm,
                vmin=min_color, vmax=max_color, marker='o', s=scatter_size)
    plt.colorbar()
    plt.title('Prediction (worst case)', fontsize=15)

    plt.subplot(2, 3, 2)
    plt.scatter(coor_x, coor_y, c=tracking['worst_gt'], cmap=cm,
                vmin=min_color, vmax=max_color, marker='o', s=scatter_size)
    plt.title('Ground Truth (worst case)', fontsize=15)
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.scatter(coor_x, coor_y, c=np.abs(tracking['worst_f'] - tracking['worst_gt']),
                cmap=cm, vmin=0, vmax=max_color, marker='o', s=scatter_size)
    plt.title('Absolute Error (worst case)', fontsize=15)
    plt.colorbar()

    # Best case row
    plt.subplot(2, 3, 4)
    plt.scatter(coor_x, coor_y, c=tracking['best_f'], cmap=cm,
                vmin=min_color, vmax=max_color, marker='o', s=scatter_size)
    plt.colorbar()
    plt.title('Prediction (best case)', fontsize=15)

    plt.subplot(2, 3, 5)
    plt.scatter(coor_x, coor_y, c=tracking['best_gt'], cmap=cm,
                vmin=min_color, vmax=max_color, marker='o', s=scatter_size)
    plt.title('Ground Truth (best case)', fontsize=15)
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.scatter(coor_x, coor_y, c=np.abs(tracking['best_f'] - tracking['best_gt']),
                cmap=cm, vmin=0, vmax=max_color, marker='o', s=scatter_size)
    plt.title('Absolute Error (best case)', fontsize=15)
    plt.colorbar()


def save_test_plot(args):
    """
    Save the test plot with appropriate filename.

    Args:
        args: Arguments containing model and data names

    Returns:
        None (saves the figure)
    """
    plt.savefig(r'../res/plots/sample_{}_{}.png'.format(args.model, args.data))


def run_test_loop(model, loader, test_coor_x, test_coor_y, device,
                  compute_L2_error, num_outputs=1):
    """
    Generic test loop that iterates over data loader and computes errors.

    Args:
        model: Model to test
        loader: Data loader
        test_coor_x: Test x-coordinates (1, M)
        test_coor_y: Test y-coordinates (1, M)
        device: torch device
        compute_L2_error: Function that computes L2 error (pred, out) -> L2_relative
        num_outputs: Number of model outputs (1 for single, 2 for multi-output)

    Returns:
        tuple: (mean_relative_L2, tracking_dict_with_best_worst)
    """
    tracking = initialize_error_tracking()

    for batch_data in loader:
        if num_outputs == 1:
            par, out = batch_data
            par = par.float().to(device)
            out = out.float().to(device)
            batch = par.shape[0]

            pred = model(test_coor_x.repeat(batch, 1),
                        test_coor_y.repeat(batch, 1), par)
            L2_relative = compute_L2_error(pred, out)

        else:  # num_outputs == 2
            par, u, v = batch_data
            par = par.float().to(device)
            u = u.float().to(device)
            v = v.float().to(device)
            batch = par.shape[0]

            u_pred, v_pred = model(test_coor_x.repeat(batch, 1),
                                  test_coor_y.repeat(batch, 1), par)
            L2_relative = compute_L2_error((u_pred, v_pred), (u, v))
            pred = (u_pred, v_pred)
            out = (u, v)

        tracking = update_best_worst_cases(L2_relative, pred, out, tracking, num_outputs)
        tracking = accumulate_error(L2_relative, tracking)
        tracking['num'] += batch

    mean_relative_L2 = finalize_error_statistics(tracking, tracking['num'])
    return mean_relative_L2, tracking
