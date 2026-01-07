import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
from train_utils import (
    print_training_config,
    setup_optimizer_and_loss,
    try_load_pretrained_model,
    validate_and_save_model_generic
)

# Define physics-informed loss loss
def darcy_loss(u, x_coor, y_coor):
    '''
    PDE residual = u_xx + u_yy + 10, where 10 is the constant uniform forcing term
    '''

    # define loss
    mse = nn.MSELoss()

    # compute pde residual
    u_x = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_yy = torch.autograd.grad(outputs=u_y, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    pde_residual = u_xx + u_yy + 10
    pde_loss = mse(pde_residual, torch.zeros_like(pde_residual))

    return pde_loss

# define a function for visualization of the predicted function over the domain
def plot(xcoor, ycoor, f):

    # Create a scatter plot with color mapped to the 'f' values
    # you can change the plotting setting here
    plt.scatter(xcoor, ycoor, c=f, cmap='viridis', marker='o', s=10)

    # Add a colorbar
    plt.colorbar(label='f')

# define the function for model testing
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

    # split the coordinate into (x,y)
    test_coor_x, test_coor_y = prepare_test_coordinates(coors, device)

    # define L2 error computation for Darcy flow
    def compute_L2_error_darcy(pred, out):
        return (torch.norm(pred - out, dim=-1) / torch.norm(out, dim=-1))

    # run test loop
    mean_relative_L2, tracking = run_test_loop(
        model, loader, test_coor_x, test_coor_y, device,
        compute_L2_error_darcy, num_outputs=1
    )

    # make the coordinates to numpy
    coor_x = test_coor_x[0].detach().cpu().numpy()
    coor_y = test_coor_y[0].detach().cpu().numpy()

    # create visualization
    create_test_visualization(coor_x, coor_y, tracking, scatter_size=5)

    # save plot
    save_test_plot(args)

    return mean_relative_L2

# define the function for model validation
def val(model, loader, coors, device):
    '''
    Input:
        model: the model instance to be tested
        loader: validation loader of the dataset
        coors: A set of fixed coordinate
        device: cpu or gpu
    Ouput:
        mean_relative_L2: average relative error of the model prediction
    '''

    # split the coordinate into (x,y)
    test_coor_x = coors[:, 0].unsqueeze(0).float().to(device)
    test_coor_y = coors[:, 1].unsqueeze(0).float().to(device)

    # model testing
    mean_relative_L2 = 0
    num = 0
    for (par, out) in loader:

        # move the batch data to device
        batch = par.shape[0]
        par = par.float().to(device)
        out = out.float().to(device)

        # model forward
        pred = model(test_coor_x.repeat(batch,1), test_coor_y.repeat(batch,1), par)
        L2_relative = (torch.norm(pred-out, dim=-1) / torch.norm(out, dim=-1))
        mean_relative_L2 += torch.sum(L2_relative)
        num += par.shape[0]

    mean_relative_L2 /= num
    mean_relative_L2 = mean_relative_L2.detach().cpu().item()

    return mean_relative_L2

def prepare_darcy_coordinates(coors, BC_flags, device):
    """
    Prepare and organize coordinates for Darcy flow training.

    Returns:
        dict: Dictionary containing organized coordinates
    """
    BC_coors = coors[np.where(BC_flags==1)[0], :].float().to(device)
    pde_coors = coors[np.where(BC_flags==0)[0], :]
    num_pde_nodes = pde_coors.shape[0]

    return {
        'BC': BC_coors,
        'pde': pde_coors,
        'num_pde_nodes': num_pde_nodes
    }


def sample_pde_coordinates_darcy(pde_coors, num_pde_nodes, sampling_size):
    """
    Sample PDE collocation points for Darcy flow (uniform sampling).

    Returns:
        torch.Tensor: Sampled PDE coordinates
    """
    ss_index = np.random.randint(0, num_pde_nodes, sampling_size)
    return pde_coors[ss_index, :]


def train_single_iteration_darcy(model, par, out, prepared_coords, BC_flags, mse,
                                 bc_weight, optimizer, device, coor_sampling_size):
    """
    Perform a single training iteration for Darcy flow.

    Returns:
        dict: Loss values for this iteration
    """
    # Sample PDE coordinates
    pde_sampled_coors = sample_pde_coordinates_darcy(
        prepared_coords['pde'],
        prepared_coords['num_pde_nodes'],
        coor_sampling_size
    )

    # Prepare data
    batch = par.shape[0]
    par = par.float().to(device)
    BC_gt = out[:, np.where(BC_flags==1)[0]].float().to(device)
    pde_sampled_coors_r = pde_sampled_coors.unsqueeze(0).repeat(batch, 1, 1).float().to(device)
    bc_sampled_coors_r = prepared_coords['BC'].unsqueeze(0).repeat(batch, 1, 1).float().to(device)

    # Forward pass for BC prediction
    BC_pred = model(bc_sampled_coors_r[:, :, 0], bc_sampled_coors_r[:, :, 1], par)

    # Define differentiable variables for PDE
    sampled_x_coors = Variable(pde_sampled_coors_r[:, :, 0].type(torch.FloatTensor), requires_grad=True).to(device)
    sampled_y_coors = Variable(pde_sampled_coors_r[:, :, 1].type(torch.FloatTensor), requires_grad=True).to(device)

    # Forward pass for PDE
    u_pred = model(sampled_x_coors, sampled_y_coors, par)

    # Compute losses
    pde_loss = darcy_loss(u_pred, sampled_x_coors, sampled_y_coors)
    bc_loss = mse(BC_pred, BC_gt)
    total_loss = pde_loss + bc_weight * bc_loss

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'pde': pde_loss.detach().cpu().item(),
        'bc': bc_loss.detach().cpu().item(),
        'total': total_loss
    }


# define the function for model training
def train(args, config, model, device, loaders, coors, BC_flags):
    '''
    Input:
        args: usig this information to assign name for the output plots
        config: store the configuration for model training and testing
        model: model instance to be trained
        device: cpu or gpu
        loaders: a tuple to store (train_loader, val_loader, test_loader)
        coors: A set of fixed coordinate in the shape of (M,2)
        BC_flags: A set of binary number for the boundary indicator
            - BC_flags[i] == 1 means that coors[i,:] is the coordinate on the boundary

    '''

    # print training configuration
    print_training_config(config)

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # setup training components
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # prepare coordinates
    prepared_coords = prepare_darcy_coordinates(coors, BC_flags, device)

    # setup optimizer and loss
    mse, optimizer = setup_optimizer_and_loss(config, model)

    # visual frequency for evaluation
    vf = config['train']['visual_freq']

    # move model to device
    model = model.to(device)

    # initialize recorded loss values
    avg_losses = {
        'pde': 0,
        'bc': 0
    }

    # try loading pre-trained model
    try_load_pretrained_model(model, args, device)

    # define training weight
    bc_weight = config['train']['bc_weight']

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf

        for e in pbar:

            # validation and model saving
            min_val_err, _ = validate_and_save_model_generic(
                model, val_loader, coors, device, args, e,
                min_val_err, avg_losses, vf, val
            )

            # train one epoch
            model.train()
            for (par, out) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # perform single training iteration
                    losses = train_single_iteration_darcy(
                        model, par, out, prepared_coords, BC_flags, mse,
                        bc_weight, optimizer, device, config['train']['coor_sampling_size']
                    )

                    # accumulate losses
                    avg_losses['pde'] += losses['pde']
                    avg_losses['bc'] += losses['bc']

            # Update progress bar with current losses
            pbar.set_description(f'Epoch {e}')
            pbar.set_postfix({
                'PDE': f"{avg_losses['pde']:.6f}",
                'BC': f"{avg_losses['bc']:.6f}"
            })

    # final test
    model.load_state_dict(torch.load(
        r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model)
    ))
    model.eval()
    err = test(model, test_loader, coors, device, args)
    print('Best L2 relative error on test loader:', err)