"""
Shared training utilities for PI-DCON models.

This module contains common helper functions used across different
physics-informed training tasks (Darcy flow, plate stress, etc.)
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# GradNorm: Gradient Normalization for Multi-Task Learning
# ============================================================================
# Based on: "GradNorm: Gradient Normalization for Adaptive Loss Balancing 
# in Deep Multitask Networks" (Chen et al., ICML 2018)
# https://arxiv.org/abs/1711.02257
# ============================================================================

class GradNorm:
    """
    Full GradNorm implementation faithful to the paper.
    
    Key algorithm (Algorithm 1 from paper):
    1. Weights w_i are learnable parameters updated by GradNorm loss only
    2. Weighted loss: L = sum(w_i * L_i)
    3. Gradient norms: G_i = ||w_i * ∇_W L_i||_2 for shared layer W
    4. Loss ratio: L̃_i(t) = L_i(t) / L_i(0)
    5. Relative inverse training rate: r_i(t) = L̃_i(t) / mean(L̃(t))
    6. Target gradient: Ḡ * r_i^α  (where Ḡ = mean(G_i))
    7. GradNorm loss: L_grad = sum(|G_i - Ḡ * r_i^α|)
    8. Weight renormalization: w_i <- w_i * T / sum(w_j)
    
    Args:
        num_tasks: Number of tasks/losses to balance
        alpha: Asymmetry hyperparameter (higher = more aggressive balancing)
        lr_weights: Learning rate for weight updates (default: 0.025)
        device: Target device (cpu or cuda)
    """
    
    def __init__(self, num_tasks, alpha=1.5, lr_weights=0.025, device='cpu'):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device
        self.lr_weights = lr_weights
        self.T = float(num_tasks)  # sum of weights to maintain
        
        # Initialize task weights as learnable parameters (paper: start at 1.0)
        self.weights = nn.Parameter(torch.ones(num_tasks, device=device))
        
        # Store initial losses L_i(0) for computing relative training rates
        self.initial_losses = None
    
    def get_weights_list(self):
        """Return weights as a list of floats."""
        return self.weights.detach().cpu().tolist()
    
    def get_weights(self):
        """Return weights as a detached tensor."""
        return self.weights.detach()
    
    def step(self, losses, shared_layer, model_optimizer):
        """
        Perform one GradNorm step following Algorithm 1 from the paper.
        
        The key insight from the paper: weights are updated ONLY by the GradNorm
        loss, not by the main task loss. We achieve this by:
        1. Computing weighted loss and backprop for model params
        2. Zeroing out weight gradients (they shouldn't be updated by task loss)
        3. Computing GradNorm loss and getting gradients for weights only
        4. Manually updating weights with those gradients
        
        Args:
            losses: List or tensor of individual task losses (unweighted, with grad)
            shared_layer: The shared layer W to compute gradients on
                         (typically last shared layer before task heads)
            model_optimizer: The optimizer for the main model parameters
        
        Returns:
            weighted_loss: The weighted sum of task losses (detached, for logging)
        """
        if not isinstance(losses, torch.Tensor):
            losses = torch.stack(losses)
        
        # Initialize L_i(0) on first iteration
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
        
        # Step 1: Compute weighted loss L = sum(w_i * L_i)
        weighted_task_loss = self.weights * losses
        total_loss = weighted_task_loss.sum()
        
        # Step 2: Backward pass for model parameters
        model_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        
        # Step 3: Zero out weight gradients - weights are updated by GradNorm loss only!
        if self.weights.grad is not None:
            self.weights.grad.data.zero_()
        
        # Step 4: Compute gradient norms G_i = ||w_i * ∇_W L_i||_2
        # Note: we compute grad of UNWEIGHTED loss, then multiply by weight
        W = shared_layer
        norms = []
        for i in range(self.num_tasks):
            # Get gradient of unweighted task loss w.r.t. shared layer
            gygw = torch.autograd.grad(
                losses[i], 
                W.parameters(), 
                retain_graph=True,
                create_graph=True  # Need this for GradNorm loss gradient
            )[0]
            # Multiply by weight and compute norm: G_i = ||w_i * ∇_W L_i||
            norms.append(torch.norm(self.weights[i] * gygw))
        norms = torch.stack(norms)
        
        # Step 5: Compute loss ratios L̃_i(t) = L_i(t) / L_i(0)
        loss_ratio = losses.detach() / (self.initial_losses + 1e-8)
        
        # Step 6: Relative inverse training rate r_i = L̃_i / mean(L̃)
        inverse_train_rate = loss_ratio / (loss_ratio.mean() + 1e-8)
        
        # Step 7: Average gradient norm Ḡ (detached - this is a target)
        mean_norm = norms.mean().detach()
        
        # Step 8: Compute GradNorm loss = sum(|G_i - Ḡ * r_i^α|)
        # The target Ḡ * r_i^α must be detached (constant)
        target = mean_norm * (inverse_train_rate ** self.alpha)
        gradnorm_loss = torch.abs(norms - target).sum()
        
        # Step 9: Compute gradient of GradNorm loss w.r.t. weights
        weight_grad = torch.autograd.grad(gradnorm_loss, self.weights)[0]
        
        # Step 10: Update model parameters
        model_optimizer.step()
        
        # Step 11: Update weights manually (simple gradient descent)
        with torch.no_grad():
            self.weights.data -= self.lr_weights * weight_grad
            
            # Step 12: Renormalize weights so sum = T (as per paper)
            # Also clamp to positive values
            self.weights.data = torch.clamp(self.weights.data, min=0.0)
            normalize_coeff = self.T / (self.weights.data.sum() + 1e-8)
            self.weights.data = self.weights.data * normalize_coeff
        
        return total_loss.detach()
    
    def reset(self):
        """Reset the GradNorm state."""
        self.weights = nn.Parameter(torch.ones(self.num_tasks, device=self.device))
        self.initial_losses = None


class GradNormSimple:
    """
    Simplified loss-ratio reweighting (NOT full GradNorm from the paper).
    
    This is a gradient-free heuristic that adjusts weights based on relative
    training progress. It uses the same loss-ratio formula from the paper but
    does NOT compute actual gradient norms, making it easier to integrate but
    less principled than the full algorithm.
    
    Key differences from the paper:
    - No gradient computation (uses loss ratios only)
    - No separate weight optimizer (direct update rule)
    - Uses smoothed loss history instead of instantaneous values
    
    For the full algorithm faithful to the paper, use `GradNorm.step()`.
    
    Args:
        num_tasks: Number of tasks/losses to balance
        alpha: Asymmetry hyperparameter (default: 1.5)
        window_size: Number of iterations to consider for moving average
        device: Target device
    """
    
    def __init__(self, num_tasks, alpha=1.5, window_size=50, device='cpu'):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.window_size = window_size
        self.device = device
        
        # Task weights
        self.weights = torch.ones(num_tasks, device=device)
        
        # Loss history for computing training rates
        self.loss_history = [[] for _ in range(num_tasks)]
        self.initial_losses = None
    
    def get_weights_list(self):
        """Return weights as a list of floats."""
        return self.weights.cpu().tolist()
    
    def get_weights(self):
        """Return weights as a detached tensor."""
        return self.weights.detach()
    
    def update_and_get_weighted_loss(self, losses):
        """
        Update weights based on loss values and return weighted loss.
        
        Args:
            losses: List or dict of individual task losses
        
        Returns:
            weighted_loss: The weighted sum of task losses
        """
        # Convert to list if dict
        if isinstance(losses, dict):
            losses = list(losses.values())
        
        # Convert to tensors if needed
        loss_values = []
        for l in losses:
            if isinstance(l, torch.Tensor):
                loss_values.append(l.detach().item())
            else:
                loss_values.append(float(l))
        
        # Initialize initial losses
        if self.initial_losses is None:
            self.initial_losses = loss_values.copy()
        
        # Update history
        for i, l in enumerate(loss_values):
            self.loss_history[i].append(l)
            if len(self.loss_history[i]) > self.window_size:
                self.loss_history[i].pop(0)
        
        # Compute relative inverse training rates
        # Training rate = initial_loss / current_loss (higher means faster training)
        current_losses = [sum(h[-min(10, len(h)):]) / min(10, len(h)) 
                          for h in self.loss_history]
        
        loss_ratios = []
        for i in range(self.num_tasks):
            ratio = current_losses[i] / (self.initial_losses[i] + 1e-8)
            loss_ratios.append(ratio)
        
        mean_ratio = sum(loss_ratios) / len(loss_ratios)
        
        # Relative inverse training rate
        # Tasks with higher ratio (slower to decrease) get higher weight
        relative_rates = [(r / (mean_ratio + 1e-8)) ** self.alpha for r in loss_ratios]
        
        # Update weights
        total = sum(relative_rates)
        self.weights = torch.tensor(
            [self.num_tasks * r / total for r in relative_rates],
            device=self.device
        )
        
        # Compute weighted loss
        if isinstance(losses[0], torch.Tensor):
            weighted_loss = sum(w * l for w, l in zip(self.weights, losses))
        else:
            # If losses are already detached, just return sum (for logging)
            weighted_loss = sum(w.item() * l for w, l in zip(self.weights, loss_values))
        
        return weighted_loss
    
    def reset(self):
        """Reset the GradNorm state."""
        self.weights = torch.ones(self.num_tasks, device=self.device)
        self.loss_history = [[] for _ in range(self.num_tasks)]
        self.initial_losses = None


def create_gradnorm(num_tasks, config, device='cpu'):
    """
    Factory function to create a GradNorm instance based on config.
    
    Args:
        num_tasks: Number of tasks to balance
        config: Configuration dict (should have 'train' -> 'gradnorm' settings)
        device: Target device
    
    Returns:
        GradNorm instance or None if disabled
    """
    gradnorm_config = config.get('train', {}).get('gradnorm', {})
    
    if not gradnorm_config.get('enabled', False):
        return None
    
    alpha = gradnorm_config.get('alpha', 1.5)
    lr_weights = gradnorm_config.get('lr_weights', 0.025)
    use_simple = gradnorm_config.get('use_simple', False)
    
    if use_simple:
        # Use simplified loss-ratio reweighting (gradient-free)
        window_size = gradnorm_config.get('window_size', 50)
        return GradNormSimple(
            num_tasks=num_tasks,
            alpha=alpha,
            window_size=window_size,
            device=device
        )
    else:
        # Use full GradNorm algorithm faithful to the paper
        return GradNorm(
            num_tasks=num_tasks,
            alpha=alpha,
            lr_weights=lr_weights,
            device=device
        )


def get_shared_layer(model):
    """
    Get the shared layer from a model for GradNorm gradient computation.
    
    The shared layer is typically the first layer of the branch network,
    which encodes the boundary condition information shared across all outputs.
    
    Args:
        model: Neural network model (DeepONet, DCON, etc.)
    
    Returns:
        nn.Module: The shared layer (typically first Linear layer of branch)
    """
    # Try common layer names in order of preference
    if hasattr(model, 'branch'):
        # DCON and DeepONet models have a branch network
        branch = model.branch
        if isinstance(branch, nn.Sequential):
            # First layer of Sequential is the shared layer
            return branch[0]
        elif isinstance(branch, nn.ModuleList):
            # First branch's first layer
            return branch[0][0]
    
    if hasattr(model, 'fc'):
        # Simple models with fc layer
        if isinstance(model.fc, nn.Sequential):
            return model.fc[0]
    
    # Fallback: find first Linear layer
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module
    
    raise ValueError("Could not find a shared layer in the model for GradNorm")

