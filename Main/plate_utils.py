import torch.nn as nn
import math
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from train_utils import (
    print_training_config,
    setup_optimizer_and_loss,
    try_load_pretrained_model,
    validate_and_save_model_generic
)
from gradnorm import (
    GradNorm,
    GradNormSimple,
    create_gradnorm,
    get_shared_layer
)


# Physics-informed loss
def struct_loss(u, v, x_coor, y_coor, params):

    # extract parameters
    E, mu = params
    G = E / 2 / (1+mu)

    # compute strain
    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    eps_xy = (u_y + v_x)

    # compute stress
    # sigma_xx = 2 * G * eps_xx + (E * mu / (1+mu)/ (1-2*mu)) * (eps_xx + eps_yy + eps_zz)
    # sigma_yy = 2 * G * eps_yy + (E * mu / (1+mu)/ (1-2*mu)) * (eps_xx + eps_yy + eps_zz)  # E * (eps_yy + mu * (eps_xx + eps_zz))
    # sigma_zz = 2 * G * eps_zz + (E * mu / (1+mu)/ (1-2*mu)) * (eps_xx + eps_yy + eps_zz)  # E * (eps_zz + mu * (eps_xx + eps_yy))
    sigma_xx = (E / (1-mu**2)) * (eps_xx + mu*(eps_yy))
    sigma_yy = (E / (1-mu**2)) * (eps_yy + mu*(eps_xx))
    sigma_xy = G * eps_xy

    # compute residual
    rx = torch.autograd.grad(outputs=sigma_xx, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xx),create_graph=True)[0] +\
         torch.autograd.grad(outputs=sigma_xy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_xy),create_graph=True)[0]
    ry = torch.autograd.grad(outputs=sigma_xy, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xx),create_graph=True)[0] +\
         torch.autograd.grad(outputs=sigma_yy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_xy),create_graph=True)[0]

    return rx, ry

# Neumann Boundation condition loss
def bc_edgeY_loss(u, v, x_coor, y_coor, params):

    # extract parameters
    E, mu = params
    G = E / 2 / (1+mu)

    # compute strain
    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    eps_xy = (u_y + v_x)
    
    # compute stress
    sigma_yy = (E / (1-mu**2)) * (eps_yy + mu*(eps_xx))
    sigma_xy = G * eps_xy

    return sigma_yy, sigma_xy

# function for ploting the predicted function
def plot(xcoor, ycoor, f):

    # Create a scatter plot with color mapped to the 'f' values
    plt.scatter(xcoor, ycoor, c=f, cmap='viridis', marker='o', s=15)
    # Add a colorbar
    plt.colorbar(label='f')

# function for testing
def test(model, loader, coors, device, args):
    '''
    Input:
        model: the model instance to be tested
        loader: testing loader of the dataset
        coors: A set of fixed coordinate
        device: cpu or gpu
        args: usig this information to assign name for the output plots
    Ouput:
        A plot of the PDE solution ground-truth, prediction, and absolute error
    '''
    from test_utils import (
        prepare_test_coordinates,
        run_test_loop,
        create_test_visualization,
        save_test_plot
    )

    # split the coordinates
    test_coor_x, test_coor_y = prepare_test_coordinates(coors, device)

    # define L2 error computation for plate stress
    def compute_L2_error_plate(pred, out):
        u_pred, v_pred = pred
        u, v = out
        return torch.sqrt(torch.sum((u_pred-u)**2 + (v_pred-v)**2, -1)) / \
               torch.sqrt(torch.sum((u)**2 + (v)**2, -1))

    # run test loop
    mean_relative_L2, tracking = run_test_loop(
        model, loader, test_coor_x, test_coor_y, device,
        compute_L2_error_plate, num_outputs=2
    )

    # make the coordinates to numpy
    coor_x = test_coor_x[0].detach().cpu().numpy()
    coor_y = test_coor_y[0].detach().cpu().numpy()

    # create visualization with larger scatter size for plate problem
    create_test_visualization(coor_x, coor_y, tracking, scatter_size=20)

    # save plot
    save_test_plot(args)

    return mean_relative_L2

def val(model, loader, coors, device):

    # split the coordinates
    test_coor_x = coors[:, 0].unsqueeze(0).float().to(device)
    test_coor_y = coors[:, 1].unsqueeze(0).float().to(device)

    mean_relative_L2 = 0
    num = 0
    for (par, u, v) in loader:
        
        # move the data to device
        batch = par.shape[0]
        par = par.float().to(device)
        u = u.float().to(device)
        v = v.float().to(device)

        # model forward
        u_pred, v_pred = model(test_coor_x.repeat(batch,1), test_coor_y.repeat(batch,1), par)
        L2_relative = torch.sqrt(torch.sum((u_pred-u)**2 + (v_pred-v)**2, -1)) / torch.sqrt(torch.sum((u)**2 + (v)**2, -1))

        # compute relative error
        mean_relative_L2 += torch.sum(L2_relative)
        num += u.shape[0]

        # compute absolute error for point sampling probability computation
        abs_err = torch.mean(torch.abs(u_pred-u) + torch.abs(v_pred-v), 0).detach().cpu().numpy()
        
        # # check GPU usage
        # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

    mean_relative_L2 /= num
    mean_relative_L2 = mean_relative_L2.detach().cpu().item()

    return mean_relative_L2, abs_err


def prepare_training_coordinates(coors, flag_BCxy, flag_BCy, flag_BC_load, device):
    """
    Prepare and organize different types of coordinates for training.

    Returns:
        dict: Dictionary containing organized coordinates for different boundary types
    """
    xy_BC_coors = coors[np.where(flag_BCxy==1)[0],:]
    y_BC_coors = coors[np.where(flag_BCy==1)[0],:]
    load_BC_coors = coors[np.where(flag_BC_load==1)[0],:]
    pde_coors = coors[np.where(flag_BC_load+flag_BCxy+flag_BCy==0)[0],:]
    num_pde_nodes = pde_coors.shape[0]

    # Move to device
    xy_BC_coors = xy_BC_coors.float().to(device)
    y_BC_coors = y_BC_coors.float().to(device)
    load_BC_coors = load_BC_coors.float().to(device)
    coors = coors.float().to(device)

    print('Number of PDE points:', num_pde_nodes)

    return {
        'xy_BC': xy_BC_coors,
        'y_BC': y_BC_coors,
        'load_BC': load_BC_coors,
        'pde': pde_coors,
        'all': coors,
        'num_pde_nodes': num_pde_nodes
    }


def compute_all_losses(u_load_pred, v_load_pred, u_load_gt, v_load_gt,
                      u_BCxy_pred, v_BCxy_pred, sigma_yy, sigma_xy,
                      rx_pde, ry_pde, mse, weight_bc, gradnorm=None,
                      shared_layer=None, optimizer=None):
    """
    Compute all loss components for training.

    Args:
        gradnorm: Optional GradNorm or GradNormSimple instance for adaptive loss weighting
        shared_layer: Optional shared layer for full GradNorm (required for GradNorm)
        optimizer: Optional model optimizer for full GradNorm (required for GradNorm)

    Returns:
        dict: Dictionary containing all loss components
    """
    bc_loss1 = mse(u_load_pred, u_load_gt)
    bc_loss2 = mse(v_load_pred, v_load_gt)
    bc_loss3 = torch.mean(u_BCxy_pred**2) + torch.mean(v_BCxy_pred**2)
    bc_loss4 = torch.mean(sigma_yy**2) + torch.mean(sigma_xy**2)
    pde_loss1 = torch.mean(rx_pde**2)
    pde_loss2 = torch.mean(ry_pde**2)
    
    all_losses = [bc_loss1, bc_loss2, bc_loss3, bc_loss4, pde_loss1, pde_loss2]
    
    # Compute total loss with GradNorm or fixed weights
    if isinstance(gradnorm, GradNorm):
        # Full GradNorm algorithm (paper-faithful)
        # GradNorm handles its own backward pass and weight updates
        gradnorm.step(all_losses, shared_layer, optimizer)
        # Compute weighted loss for logging (no backward needed)
        with torch.no_grad():
            weights = gradnorm.get_weights()
            total_loss = (weights * torch.stack(all_losses)).sum()
    elif isinstance(gradnorm, GradNormSimple):
        # Simplified GradNorm (gradient-free heuristic)
        total_loss = gradnorm.update_and_get_weighted_loss(all_losses)
    else:
        # Use fixed weight_bc
        total_loss = (bc_loss1 + bc_loss2 + bc_loss3 + bc_loss4) + (pde_loss1 + pde_loss2) * weight_bc

    return {
        'bc1': bc_loss1.detach().cpu().item(),
        'bc2': bc_loss2.detach().cpu().item(),
        'bc3': bc_loss3.detach().cpu().item(),
        'bc4': bc_loss4.detach().cpu().item(),
        'pde1': pde_loss1.detach().cpu().item(),
        'pde2': pde_loss2.detach().cpu().item(),
        'total': total_loss,
        'uses_full_gradnorm': isinstance(gradnorm, GradNorm)
    }


def sample_pde_coordinates(pde_coors, num_pde_nodes, pointwise_err, flags, config):
    """
    Sample PDE collocation points based on error probability.

    Returns:
        torch.Tensor: Sampled PDE coordinates
    """
    p_pde_sampling = pointwise_err[np.where(flags['BC_load']+flags['BCxy']+flags['BCy']==0)[0]]
    p_pde_sampling = p_pde_sampling / np.sum(p_pde_sampling)

    ss_index = np.random.choice(
        np.arange(num_pde_nodes),
        config['train']['coor_sampling_size'],
        p=p_pde_sampling
    )
    return pde_coors[ss_index, :]


def train_single_iteration(model, par, u, v, prepared_coords, params, mse, weight_bc,
                          optimizer, pointwise_err, flags, config, gradnorm=None,
                          shared_layer=None):
    """
    Perform a single training iteration with coordinate sampling.

    Args:
        gradnorm: Optional GradNorm or GradNormSimple instance for adaptive loss weighting
        shared_layer: Optional shared layer for full GradNorm (required for GradNorm)

    Returns:
        dict: Loss values for this iteration
    """
    device = par.device
    batchsize = par.shape[0]

    # Sample PDE coordinates
    pde_sampled_coors = sample_pde_coordinates(
        prepared_coords['pde'],
        prepared_coords['num_pde_nodes'],
        pointwise_err,
        flags,
        config
    )
    pde_sampled_coors = pde_sampled_coors.float().to(device)

    # Prepare and repeat data for batch
    par = par.float().to(device)
    pde_sampled_coors_r = pde_sampled_coors.unsqueeze(0).repeat(batchsize, 1, 1)
    xy_BC_coors_r = prepared_coords['xy_BC'].unsqueeze(0).repeat(batchsize, 1, 1)
    y_BC_coors_r = prepared_coords['y_BC'].unsqueeze(0).repeat(batchsize, 1, 1)
    load_BC_coors_r = prepared_coords['load_BC'].unsqueeze(0).repeat(batchsize, 1, 1)

    # Forward pass on fixed boundary
    u_BCxy_pred, v_BCxy_pred = model(xy_BC_coors_r[:, :, 0], xy_BC_coors_r[:, :, 1], par)

    # Forward pass on free boundary
    x_pde_bcy = Variable(y_BC_coors_r[:, :, 0], requires_grad=True)
    y_pde_bcy = Variable(y_BC_coors_r[:, :, 1], requires_grad=True)
    u_BCy_pred, v_BCy_pred = model(x_pde_bcy, y_pde_bcy, par)
    sigma_yy, sigma_xy = bc_edgeY_loss(u_BCy_pred, v_BCy_pred, x_pde_bcy, y_pde_bcy, params)

    # Forward pass on loading element
    u_load_pred, v_load_pred = model(load_BC_coors_r[:, :, 0], load_BC_coors_r[:, :, 1], par)
    u_load_gt = u[:, np.where(flags['BC_load']==1)[0]].float().to(device)
    v_load_gt = v[:, np.where(flags['BC_load']==1)[0]].float().to(device)

    # Forward pass on PDE interior
    x_pde = Variable(pde_sampled_coors_r[:, :, 0], requires_grad=True)
    y_pde = Variable(pde_sampled_coors_r[:, :, 1], requires_grad=True)
    u_pde_pred, v_pde_pred = model(x_pde, y_pde, par)
    rx_pde, ry_pde = struct_loss(u_pde_pred, v_pde_pred, x_pde, y_pde, params)

    # Compute losses
    losses = compute_all_losses(
        u_load_pred, v_load_pred, u_load_gt, v_load_gt,
        u_BCxy_pred, v_BCxy_pred, sigma_yy, sigma_xy,
        rx_pde, ry_pde, mse, weight_bc, gradnorm=gradnorm,
        shared_layer=shared_layer, optimizer=optimizer
    )

    # Backward pass and optimization (unless full GradNorm already did it)
    if not losses.get('uses_full_gradnorm', False):
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

    return losses


# define the training function
def train(args, config, model, device, loaders, coors, flag_BCxy, flag_BCy, flag_BC_load, params):

    # print training configuration
    print_training_config(config)

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # setup training components
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # prepare coordinates
    prepared_coords = prepare_training_coordinates(coors, flag_BCxy, flag_BCy, flag_BC_load, device)

    # setup optimizer and loss
    mse, optimizer = setup_optimizer_and_loss(config, model)

    # visual frequency for evaluation
    vf = config['train']['visual_freq']

    # move model to device
    model = model.to(device)

    # initialize recorded loss values
    avg_losses = {
        'bc1': 0, 'bc2': 0, 'bc3': 0, 'bc4': 0,
        'pde1': 0, 'pde2': 0
    }

    # try loading pre-trained model
    try_load_pretrained_model(model, args, device)

    # define training weight
    weight_bc = config['train']['bc_weight']

    # prepare flags dictionary
    flags = {
        'BC_load': flag_BC_load,
        'BCxy': flag_BCxy,
        'BCy': flag_BCy
    }

    # setup GradNorm if enabled (6 tasks: bc1, bc2, bc3, bc4, pde1, pde2)
    gradnorm = create_gradnorm(num_tasks=6, config=config, device=device)
    
    # Get shared layer for full GradNorm algorithm
    shared_layer = get_shared_layer(model) if isinstance(gradnorm, GradNorm) else None

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        pointwise_err = np.zeros(prepared_coords['all'].shape[0])

        for e in pbar:

            # validation and model saving
            min_val_err, pointwise_err = validate_and_save_model_generic(
                model, val_loader, prepared_coords['all'], device, args, e,
                min_val_err, avg_losses, vf, val
            )

            # train one epoch
            model.train()
            epoch_losses = {
                'bc1': 0, 'bc2': 0, 'bc3': 0, 'bc4': 0,
                'pde1': 0, 'pde2': 0
            }
            num_iterations = 0
            
            for (par, u, v) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # perform single training iteration
                    losses = train_single_iteration(
                        model, par, u, v, prepared_coords, params, mse, weight_bc,
                        optimizer, pointwise_err, flags, config, gradnorm=gradnorm,
                        shared_layer=shared_layer
                    )

                    # accumulate losses for this epoch
                    for key in epoch_losses:
                        epoch_losses[key] += losses[key]
                    num_iterations += 1
                    
                    # accumulate losses for validation period
                    avg_losses['bc1'] += losses['bc1']
                    avg_losses['bc2'] += losses['bc2']
                    avg_losses['bc3'] += losses['bc3']
                    avg_losses['bc4'] += losses['bc4']
                    avg_losses['pde1'] += losses['pde1']
                    avg_losses['pde2'] += losses['pde2']

            # Update progress bar with current epoch average losses (unweighted true values)
            pbar.set_description(f'Epoch {e}')
            postfix_dict = {
                'PDE(raw)': f"{(epoch_losses['pde1'] + epoch_losses['pde2']) / num_iterations:.6f}",
                'BC(raw)': f"{(epoch_losses['bc1'] + epoch_losses['bc2'] + epoch_losses['bc3'] + epoch_losses['bc4']) / num_iterations:.6f}"
            }
            # Show GradNorm weights if enabled (show aggregated BC and PDE weights)
            if gradnorm is not None:
                weights = gradnorm.get_weights_list()
                postfix_dict['w_bc'] = f"{sum(weights[:4]):.2f}"
                postfix_dict['w_pde'] = f"{sum(weights[4:]):.2f}"
            pbar.set_postfix(postfix_dict)

    # final test
    model.load_state_dict(torch.load(
        r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model)
    ))
    model.eval()
    err = test(model, test_loader, prepared_coords['all'], device, args)
    print('Best L2 relative error on test loader:', err)