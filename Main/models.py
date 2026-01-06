import torch
import torch.nn as nn

'''
Neural operator models for 2D Darcy flow and plate stress problems
Refactored with class hierarchy to reduce code duplication
'''

# ============================================================================
# Utility Functions
# ============================================================================

def build_fc_block(in_dim, out_dim, activation=nn.Tanh()):
    """Build a single fully-connected block with activation"""
    return [nn.Linear(in_dim, out_dim), activation]


def build_mlp(in_dim, fc_dim, n_layers, out_dim=None, activation=nn.Tanh()):
    """
    Build a multi-layer perceptron

    Args:
        in_dim: Input dimension
        fc_dim: Hidden layer dimension
        n_layers: Number of layers
        out_dim: Output dimension (defaults to fc_dim if None)
        activation: Activation function

    Returns:
        List of layers suitable for nn.Sequential
    """
    if out_dim is None:
        out_dim = fc_dim

    layers = build_fc_block(in_dim, fc_dim, activation)
    for _ in range(n_layers - 1):
        layers.extend(build_fc_block(fc_dim, fc_dim, activation))
    layers.append(nn.Linear(fc_dim, out_dim))

    return layers


class fc(nn.Module):
    """FC layer wrapper used in plate models"""
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.ff(x))


# ============================================================================
# Base Architecture Classes
# ============================================================================

class BaseNeuralOperator(nn.Module):
    """Base class for all neural operator models"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc_dim = config['model']['fc_dim']
        self.n_layer = config['model']['N_layer']


class BaseDeepONet(BaseNeuralOperator):
    """
    Base class for DeepONet architecture.
    Shared logic for both Darcy and Plate problems.
    """
    def __init__(self, config, input_dim):
        super().__init__(config)

        # Branch network - encodes boundary condition function values
        branch_layers = build_mlp(input_dim, self.fc_dim, self.n_layer, self.fc_dim)
        self.branch = nn.Sequential(*branch_layers)

        # Trunk network - encodes spatial coordinates
        trunk_layers = build_mlp(2, self.fc_dim, self.n_layer, self.fc_dim)
        self.trunk = nn.Sequential(*trunk_layers)


class BaseImprovedDeepONet(BaseNeuralOperator):
    """
    Base class for Improved DeepONet with embedding modulation.
    Shared logic for both Darcy and Plate problems.
    """
    def __init__(self, config, input_dim):
        super().__init__(config)

        # Branch network layers
        self.FC1b = nn.Linear(input_dim, self.fc_dim)
        self.FC2b = nn.Linear(self.fc_dim, self.fc_dim)
        self.FC3b = nn.Linear(self.fc_dim, self.fc_dim)

        # Trunk network layers
        self.FC1c = nn.Linear(2, self.fc_dim)
        self.FC2c = nn.Linear(self.fc_dim, self.fc_dim)
        self.FC3c = nn.Linear(self.fc_dim, self.fc_dim)

        # Embedding networks
        self.be = nn.Sequential(nn.Linear(input_dim, self.fc_dim), nn.Tanh())
        self.ce = nn.Sequential(nn.Linear(2, self.fc_dim), nn.Tanh())

        # Activation
        self.act = nn.Tanh()


class BaseDCON(BaseNeuralOperator):
    """
    Base class for DCON architecture with max-pooling and gated modulation.
    Shared logic for both Darcy and Plate problems.
    """
    def __init__(self, config):
        super().__init__(config)

        # Branch network - processes (x, y, u) tuples
        branch_layers = build_mlp(3, self.fc_dim, self.n_layer, self.fc_dim)
        self.branch = nn.Sequential(*branch_layers)

        # Trunk network with gated modulation
        self.FC1u = nn.Linear(2, self.fc_dim)
        self.FC2u = nn.Linear(self.fc_dim, self.fc_dim)
        self.FC3u = nn.Linear(self.fc_dim, self.fc_dim)

        # Activation
        self.act = nn.Tanh()


# ============================================================================
# Darcy Flow Problem Models
# ============================================================================

class DeepONet_darcy(BaseDeepONet):
    """DeepONet for Darcy flow problem"""

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        # Only extract the function values information to represent the PDE parameters
        enc = self.branch(par[...,-1])

        # Compute the PDE solution prediction
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)
        x = self.trunk(xy)    # (B,M,F)
        u = torch.einsum('bij,bj->bi', x, enc)

        return u


class Improved_DeepOnet_darcy(BaseImprovedDeepONet):
    """Improved DeepONet with embedding modulation for Darcy flow"""

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        # Get the coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Get the embeddings
        par_emb = self.be(par[...,-1]).unsqueeze(1)
        coor_emb = self.ce(xy)

        # Parameter forward
        enc = self.FC1b(par[...,-1]).unsqueeze(1)
        enc = self.act(enc)
        enc = (1-enc) * par_emb + enc * coor_emb
        enc = self.FC2b(enc)
        enc = self.act(enc)
        enc = (1-enc) * par_emb + enc * coor_emb
        enc = self.FC3b(enc)

        # Coordinate forward
        xy = self.FC1c(xy)
        xy = self.act(xy)
        xy = (1-xy) * par_emb + xy * coor_emb
        xy = self.FC2c(xy)
        xy = self.act(xy)
        xy = (1-xy) * par_emb + xy * coor_emb
        xy = self.FC3c(xy)

        # Combine
        u = torch.sum(xy*enc, -1)

        return u


class DCON_darcy(BaseDCON):
    """DCON for Darcy flow problem"""

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        # Get the kernel using both the coordinate and function values information
        enc = self.branch(par)
        enc = torch.amax(enc, 1, keepdim=True)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Predict u
        u = self.FC1u(xy)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)
        u = torch.mean(u * enc, -1)

        return u


class New_model_darcy(nn.Module):
    """Simple template model for Darcy flow"""
    def __init__(self, config):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2,128), nn.Tanh(),
            nn.Linear(128,1)
        )

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        u = self.fc(torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)).squeeze(-1)

        return u


# ============================================================================
# Plate Stress Problem Models
# ============================================================================

class DeepONet_plate(BaseDeepONet):
    """DeepONet for plate stress problem"""

    def __init__(self, config, num_input_dim):
        # Use fc layers for plate models (different from standard MLP)
        super().__init__(config)

        # Build branch networks (2 branches for u and v)
        branch_layers = [nn.Linear(num_input_dim, self.fc_dim), nn.Tanh()]
        for _ in range(self.n_layer - 1):
            branch_layers.append(fc(self.fc_dim))
        branch_layers.append(nn.Linear(self.fc_dim, self.fc_dim))
        self.branch1 = nn.Sequential(*branch_layers)
        self.branch2 = nn.Sequential(*branch_layers)

        # Build trunk networks (2 trunks for u and v)
        trunk_layers = [nn.Linear(2, self.fc_dim), nn.Tanh()]
        for _ in range(self.n_layer - 1):
            trunk_layers.append(fc(self.fc_dim))
        trunk_layers.append(nn.Linear(self.fc_dim, self.fc_dim))
        self.trunk1 = nn.Sequential(*trunk_layers)
        self.trunk2 = nn.Sequential(*trunk_layers)

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): PDE solution fucntion values over the whole domain
            v (B, M): PDE solution fucntion values over the whole domain
        '''

        # PDE parameter encoding
        enc1 = self.branch1(par[:,:,-1])
        enc2 = self.branch2(par[:,:,-1])

        # PDE solution prediction
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)
        ux = self.trunk1(xy)
        uy = self.trunk2(xy)
        u = torch.einsum('bij,bj->bi', ux, enc1)
        v = torch.einsum('bij,bj->bi', uy, enc2)

        return u, v


class Improved_DeepONet_plate(BaseImprovedDeepONet):
    """Improved DeepONet with embedding modulation for plate stress"""

    def __init__(self, config, num_input_dim):
        super().__init__(config, num_input_dim)

        # Additional trunk network for v displacement
        self.FC1c1 = nn.Linear(2, self.fc_dim)
        self.FC2c1 = nn.Linear(self.fc_dim, self.fc_dim)
        self.FC3c1 = nn.Linear(self.fc_dim, self.fc_dim)

        self.FC1c2 = nn.Linear(2, self.fc_dim)
        self.FC2c2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.FC3c2 = nn.Linear(self.fc_dim, self.fc_dim)

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): PDE solution fucntion values over the whole domain
            v (B, M): PDE solution fucntion values over the whole domain
        '''

        # Get the coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Get the embeddings
        par_emb = self.be(par[...,-1]).unsqueeze(1)  # (B,1,F)
        coor_emb = self.ce(xy)  # (B, M, F)

        # Parameter forward
        enc = self.FC1b(par[...,-1]).unsqueeze(1)  # (B,F)
        enc = self.act(enc)
        enc = (1-enc) * par_emb + enc * coor_emb
        enc = self.FC2b(enc)
        enc = self.act(enc)
        enc = (1-enc) * par_emb + enc * coor_emb
        enc = self.FC3b(enc)

        # u coordinate forward
        xy1 = self.FC1c1(xy)  # (B,M,F)
        xy1 = self.act(xy1)
        xy1 = (1-xy1) * par_emb + xy1 * coor_emb
        xy1 = self.FC2c1(xy1)
        xy1 = self.act(xy1)
        xy1 = (1-xy1) * par_emb + xy1 * coor_emb
        xy1 = self.FC3c1(xy1)
        # Combine
        u = torch.sum(xy1*enc, -1)  # (B,M)

        # v coordinate forward
        xy2 = self.FC1c2(xy)  # (B,M,F)
        xy2 = self.act(xy2)
        xy2 = (1-xy2) * par_emb + xy2 * coor_emb
        xy2 = self.FC2c2(xy2)
        xy2 = self.act(xy2)
        xy2 = (1-xy2) * par_emb + xy2 * coor_emb
        xy2 = self.FC3c2(xy2)
        # Combine
        v = torch.sum(xy2*enc, -1)  # (B,M)

        return u, v


class DCON_plate(BaseNeuralOperator):
    """DCON for plate stress problem"""

    def __init__(self, config):
        super().__init__(config)

        # Branch network
        trunk_layers = build_mlp(3, self.fc_dim, self.n_layer, self.fc_dim)
        self.branch = nn.Sequential(*trunk_layers)

        # Lift layers (specific to plate model)
        self.lift = nn.Linear(3, self.fc_dim)
        self.val_lift = nn.Linear(1, self.fc_dim)

        # Trunk network 1 (for u)
        self.FC1u = nn.Linear(2, self.fc_dim)
        self.FC2u = nn.Linear(self.fc_dim, self.fc_dim)
        self.FC3u = nn.Linear(self.fc_dim, self.fc_dim)

        # Trunk network 2 (for v)
        self.FC1v = nn.Linear(2, self.fc_dim)
        self.FC2v = nn.Linear(self.fc_dim, self.fc_dim)
        self.FC3v = nn.Linear(self.fc_dim, self.fc_dim)

        self.act = nn.Tanh()

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): PDE solution fucntion values over the whole domain
            v (B, M): PDE solution fucntion values over the whole domain
        '''

        # Get the kernel
        enc = self.branch(par)
        enc = torch.amax(enc, 1, keepdim=True)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Predict u
        u = self.FC1u(xy)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)
        u = torch.mean(u * enc, -1)  # (B, M)

        # Predict v
        v = self.FC1v(xy)
        v = self.act(v)
        v = v * enc
        v = self.FC2v(v)
        v = self.act(v)
        v = v * enc
        v = self.FC3v(v)
        v = torch.mean(v * enc, -1)  # (B, M)

        return u, v


class New_model_plate(nn.Module):
    """Simple template model for plate stress"""
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(2,128), nn.Tanh(),
            nn.Linear(128,1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2,128), nn.Tanh(),
            nn.Linear(128,1)
        )

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): PDE solution fucntion values over the whole domain
            v (B, M): PDE solution fucntion values over the whole domain
        '''

        u = self.fc1(torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)).squeeze(-1)
        v = self.fc2(torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)).squeeze(-1)

        return u, v
