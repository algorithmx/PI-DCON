import torch
import torch.nn as nn
from collections import OrderedDict

'''
Neural operator models for 2D Darcy flow and plate problems.

- Darcy: predicts a scalar field u(x, y)
- Plate: predicts a 2-component vector field (u(x, y), v(x, y))

Refactored with class hierarchy to reduce code duplication.
'''

# ============================================================================
# Utility Functions
# ============================================================================

def build_mlp(
    in_dim,
    fc_dim,
    n_layers,
    out_dim=None,
    activation=nn.Tanh,
    include_output_layer=True,
    hidden_block="linear",
):
    """
    Build a multi-layer perceptron

    Args:
        in_dim: Input dimension
        fc_dim: Hidden layer dimension
        n_layers: Number of layers
        out_dim: Output dimension (defaults to fc_dim if None)
        activation: Activation module class (e.g. nn.Tanh)
        include_output_layer: If True, append final Linear(fc_dim -> out_dim)
                hidden_block: "linear" (default) or "ff".
                        - "linear": hidden layers are Linear(fc_dim, fc_dim) + activation()
                        - "ff": hidden layers are a single module with submodules named "ff" and "act"
                            to match the original `models.bak.py` plate `fc` block state_dict keys.

    Returns:
        List of layers suitable for nn.Sequential
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if out_dim is None:
        out_dim = fc_dim

    layers = [nn.Linear(in_dim, fc_dim), activation()]
    for _ in range(n_layers - 1):
        if hidden_block == "linear":
            layers.extend([nn.Linear(fc_dim, fc_dim), activation()])
        elif hidden_block == "ff":
            layers.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("ff", nn.Linear(fc_dim, fc_dim)),
                            ("act", activation()),
                        ]
                    )
                )
            )
        else:
            raise ValueError(f"hidden_block must be 'linear' or 'ff', got {hidden_block!r}")
    if include_output_layer:
        layers.append(nn.Linear(fc_dim, out_dim))

    return layers


def build_sequential_layers(in_dim, hidden_dim, n_layers):
    """
    Build a sequence of Linear layers for manual modulation networks.
    Returns individual layers as a list [FC1, FC2, FC3, ...]
    
    Args:
        in_dim: Input dimension
        hidden_dim: Hidden layer dimension
        n_layers: Number of layers (typically 3)
    
    Returns:
        List of nn.Linear layers
    """
    layers = [nn.Linear(in_dim, hidden_dim)]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
    return layers


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
    def __init__(self, field_dim, config, input_dim, hidden_block="linear"):
        super().__init__(config)

        # Branch network - encodes boundary condition function values
        branches = []
        # Trunk network - encodes spatial coordinates
        trunks = []
        for _ in range(field_dim):
            branch_layers = build_mlp(input_dim, self.fc_dim, self.n_layer, self.fc_dim, hidden_block=hidden_block)
            branches.append(nn.Sequential(*branch_layers))
            trunk_layers = build_mlp(2, self.fc_dim, self.n_layer, self.fc_dim, hidden_block=hidden_block)
            trunks.append(nn.Sequential(*trunk_layers))
        self.branch = nn.ModuleList(branches)
        self.trunk = nn.ModuleList(trunks)
    
    def _zip(self, xy, par, axis):
        # Only extract the function values information to represent the PDE parameters
        enc = self.branch[axis](par[...,-1])
        # Compute the PDE solution prediction
        x = self.trunk[axis](xy)    # (B,M,F)
        # zip
        u = torch.einsum('bij,bj->bi', x, enc)
        return u


class BlendMixin:
    """Shared primitives for mix-in based forward helpers."""

    def _linear_act(self, linear, x):
        x = linear(x)
        x = self.act(x)
        return x


class ImprovedBlendMixin(BlendMixin):
    """Shared forward helpers for Improved DeepONet-style embedding interpolation."""

    @staticmethod
    def _blend(par_emb, coor_emb, gate):
        return (1 - gate) * par_emb + gate * coor_emb

    def _linear_act_blend(self, linear, x, par_emb, coor_emb):
        gate = self._linear_act(linear, x)
        return self._blend(par_emb, coor_emb, gate)

    def _predict_head(self, x, par_emb, coor_emb, fc1, fc2, fc3):
        x = self._linear_act_blend(fc1, x, par_emb, coor_emb)
        x = self._linear_act_blend(fc2, x, par_emb, coor_emb)
        x = fc3(x)
        return x


class BaseImprovedDeepONet(ImprovedBlendMixin, BaseNeuralOperator):
    """
    Base class for Improved DeepONet with embedding modulation.
    Shared logic for both Darcy and Plate problems.
    """
    def __init__(self, field_dim, config, input_dim):
        super().__init__(config)
        if field_dim not in (1, 2):
            raise ValueError(f"field_dim must be 1 or 2, got {field_dim}")

        # Branch network layers (3 layers for modulation)
        branch_layers = build_sequential_layers(input_dim, self.fc_dim, 3)
        self.FC1b, self.FC2b, self.FC3b = branch_layers

        # Embedding networks
        self.be = nn.Sequential(*build_mlp(input_dim, self.fc_dim, 1, include_output_layer=False))
        self.ce = nn.Sequential(*build_mlp(2, self.fc_dim, 1, include_output_layer=False))

        # Trunk network layers (3 layers for coordinate encoding)
        # Use ModuleList to ensure parameters are registered
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2, self.fc_dim, 3))
            for _ in range(field_dim)
        ])

        # Activation
        self.act = nn.Tanh()

    def _encode_par(self, par_val, par_emb, coor_emb):
        """Compute the parameter encoding (shared across all field components).

        Args:
            par_val: Parameter function values (B, 1) for encoding
            par_emb: Parameter embedding (B, 1, F)
            coor_emb: Coordinate embedding (B, 1, F) - note: uses (B,1,F) from par_emb shape

        Returns:
            Parameter encoding (B, 1, F)
        """
        return self._predict_head(par_val, par_emb, coor_emb, self.FC1b, self.FC2b, self.FC3b)

    def _predict_trunk(self, xy, par_emb, coor_emb, axis):
        """Predict one field component using the trunk network at given axis."""
        fc1, fc2, fc3 = self.FC[axis]
        return self._predict_head(xy, par_emb, coor_emb, fc1, fc2, fc3)

    def _zip(self, xy, par_val, par_emb, coor_emb, axis, enc=None):
        """Combine trunk output with parameter encoding at given axis.

        Args:
            xy: Concatenated coordinates (B, M, 2)
            par_val: Parameter function values (B, 1) for encoding
            par_emb: Parameter embedding (B, 1, F)
            coor_emb: Coordinate embedding (B, M, F)
            axis: Field component index
            enc: Pre-computed parameter encoding (optional). If None, will be computed.

        Returns:
            Field component values (B, M)
        """
        if enc is None:
            enc = self._encode_par(par_val, par_emb, coor_emb)
        # Coordinate forward
        xy = self._predict_trunk(xy, par_emb, coor_emb, axis)
        # Combine
        return torch.sum(xy * enc, -1)


class DCONBlendMixin(BlendMixin):
    """Shared forward helpers for DCON-style gated modulation."""

    @staticmethod
    def _blend(x, enc):
        return x * enc

    def _linear_act_blend(self, linear, x, enc):
        x = self._linear_act(linear, x)
        return self._blend(x, enc)

    def _predict_head(self, inp, enc, fcs, *, final_gate=True, reduce_mean=True, gate_layers=None):
        """Apply gated-modulation trunk head.

        Matches the PI-GANO `DCONBlendMixin` signature, while preserving
        PI-DCON behavior when called with the default arguments.

        Args:
            inp: (B, M, Din)
            enc: (B, 1, F)
            fcs: sequence of Linear layers, length >= 2
            final_gate: if True, apply a final multiplicative gate with `enc`
            gate_layers: controls which intermediate layers apply the gate.
                - None: gate after every pre-final layer (default)
                - int k: gate only after the first k pre-final layers
            reduce_mean: if True, return mean over feature dim (B, M)
        """
        if len(fcs) < 2:
            raise ValueError(f"fcs must have >=2 layers, got {len(fcs)}")

        if gate_layers is None:
            gate_layers = len(fcs) - 1

        x = inp
        for idx, fc in enumerate(fcs[:-1]):
            x = self._linear_act(fc, x)
            if idx < gate_layers:
                x = self._blend(x, enc)

        x = fcs[-1](x)
        if final_gate:
            x = self._blend(x, enc)
        if reduce_mean:
            return torch.mean(x, -1)
        return x


class BaseDCON(DCONBlendMixin, BaseNeuralOperator):
    """
    Base class for DCON architecture with max-pooling and gated modulation.
    Shared logic for both Darcy and Plate problems.
    """
    def __init__(
        self,
        field_dim,
        config,
        par_dim=3,
    ):
        super().__init__(config)
        if field_dim not in (1, 2):
            raise ValueError(f"field_dim must be 1 or 2, got {field_dim}")

        self.field_dim = field_dim
        self.par_dim = par_dim

        # Branch network (shared)
        self.branch = nn.Sequential(*build_mlp(par_dim, self.fc_dim, self.n_layer, self.fc_dim))

        # Trunk network, general for scalar / vector fields
        # Use ModuleList to ensure parameters are registered
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2, self.fc_dim, 3))
            for _ in range(field_dim)
        ])

        # Activation
        self.act = nn.Tanh()

    def _encode_par(self, par, par_flag=None):
        """Get the kernel encoding from branch network (shared across all field components).

        Args:
            par: Boundary parameters (B, N, par_dim)
            par_flag: Optional valid flags (B, N). If provided, masked max-pooling is used.

        Returns:
            Max-pooled encoding (B, 1, F)
        """
        enc = self.branch(par)
        if par_flag is not None:
            enc = enc * par_flag.unsqueeze(-1)
        return torch.amax(enc, 1, keepdim=True)

    def _zip(self, xy, par, axis=0, enc=None, par_flag=None):
        """Get the kernel and predict field component at given axis.

        Args:
            xy: Concatenated coordinates (B, M, 2)
            par: Boundary parameters (B, N, par_dim)
            axis: Field component index
            enc: Pre-computed encoding (optional). If None, will be computed.
            par_flag: Optional valid flags (B, N). If provided, masked max-pooling is used.

        Returns:
            Field component values (B, M)
        """
        if enc is None:
            enc = self._encode_par(par, par_flag)
        # Predict field component
        return self._predict_head(xy, enc, self.FC[axis])


# ============================================================================
# Darcy Flow Problem Models, field_dim=1
# ============================================================================

class DeepONet_darcy(BaseDeepONet):
    """DeepONet for Darcy flow problem"""
    def __init__(self, config, num_input_dim, hidden_block="linear"):
        super().__init__(1, config, num_input_dim, hidden_block=hidden_block)

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): scalar field values over the domain
        '''
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)
        u = self._zip(xy, par, 0)
        return u


class Improved_DeepOnet_darcy(BaseImprovedDeepONet):
    """Improved DeepONet with embedding modulation for Darcy flow"""

    def __init__(self, config, input_dim):
        super().__init__(1, config, input_dim)

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): scalar field values over the domain
        '''
        # Get the coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Get the embeddings
        par_emb = self.be(par[...,-1]).unsqueeze(1)
        coor_emb = self.ce(xy)

        # Compute using the unified _zip method
        u = self._zip(xy, par[...,-1].unsqueeze(1), par_emb, coor_emb, 0)

        return u


class DCON_darcy(BaseDCON):
    """DCON for Darcy flow problem"""

    def __init__(self, config):
        super().__init__(1, config) # field_dim=1 for darcy / scalar-field


    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): scalar field values over the domain
        '''
        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Compute using the unified _zip method
        u = self._zip(xy, par, 0)

        return u


class New_model_darcy(nn.Module):
    """Simple template model for Darcy flow"""
    def __init__(self, config):
        super().__init__()

        # Simple 1-layer MLP: 2 -> 128 -> 1
        self.fc = nn.Sequential(*build_mlp(2, 128, 1, out_dim=1))

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): scalar field values over the domain
        '''

        u = self.fc(torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)).squeeze(-1)

        return u


# ============================================================================
# Plate Stress Problem Models, field_dim=2
# ============================================================================

class DeepONet_plate(BaseDeepONet):
    """DeepONet for plate stress problem"""
    def __init__(self, config, num_input_dim, hidden_block="ff"):
        super().__init__(2, config, num_input_dim, hidden_block=hidden_block)

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): x-component of the vector field over the domain
            v (B, M): y-component of the vector field over the domain
        '''
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)
        u = self._zip(xy, par, 0)
        v = self._zip(xy, par, 1)
        return u, v


class Improved_DeepONet_plate(BaseImprovedDeepONet):
    """Improved DeepONet with embedding modulation for plate stress"""

    def __init__(self, config, num_input_dim):
        super().__init__(2, config, num_input_dim)

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): x-component of the vector field over the domain
            v (B, M): y-component of the vector field over the domain
        '''
        # Get the coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Get the embeddings
        par_emb = self.be(par[...,-1]).unsqueeze(1)  # (B,1,F)
        coor_emb = self.ce(xy)  # (B, M, F)

        # Parameter forward (compute once, shared by both components)
        enc = self._encode_par(par[...,-1].unsqueeze(1), par_emb, coor_emb)

        # Compute u and v using the unified _zip method with shared encoding
        u = self._zip(xy, par[...,-1].unsqueeze(1), par_emb, coor_emb, 0, enc)
        v = self._zip(xy, par[...,-1].unsqueeze(1), par_emb, coor_emb, 1, enc)

        return u, v


class DCON_plate(BaseDCON):
    """DCON for plate stress problem"""

    def __init__(self, config):
        super().__init__(2, config) # field_dim=2 for plate / vector-field

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): x-component of the vector field over the domain
            v (B, M): y-component of the vector field over the domain
        '''
        # Get the kernel (compute once, shared by both components)
        enc = self._encode_par(par)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Predict field components using the unified _zip method with shared encoding
        u = self._zip(xy, par, 0, enc)
        v = self._zip(xy, par, 1, enc)

        return u, v


class New_model_plate(nn.Module):
    """Simple template model for plate stress"""
    def __init__(self, config):
        super().__init__()

        # Simple 1-layer MLPs: 2 -> 128 -> 1
        self.fc1 = nn.Sequential(*build_mlp(2, 128, 1, out_dim=1))
        self.fc2 = nn.Sequential(*build_mlp(2, 128, 1, out_dim=1))

    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points

        model output:
            u (B, M): x-component of the vector field over the domain
            v (B, M): y-component of the vector field over the domain
        '''

        u = self.fc1(torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)).squeeze(-1)
        v = self.fc2(torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)).squeeze(-1)

        return u, v
