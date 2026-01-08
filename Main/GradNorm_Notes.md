# GradNorm Implementation Notes

## Overview

GradNorm is a gradient normalization technique for adaptive loss balancing in multi-task learning, based on the paper:

> "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (Chen et al., ICML 2018)  
> https://arxiv.org/abs/1711.02257

## Algorithm Summary

Given multiple task losses $L_i$, GradNorm dynamically adjusts task weights $w_i$ so that all tasks train at similar rates.

### Key Steps (Algorithm 1 from paper)

1. **Weighted Loss**: $L = \sum_i w_i \cdot L_i$
2. **Gradient Norms**: $G_i = \|w_i \cdot \nabla_W L_i\|_2$ for shared layer $W$
3. **Loss Ratio**: $\tilde{L}_i(t) = L_i(t) / L_i(0)$
4. **Relative Inverse Training Rate**: $r_i(t) = \tilde{L}_i(t) / \text{mean}(\tilde{L}(t))$
5. **GradNorm Loss**: $L_{\text{grad}} = \sum_i |G_i - \bar{G} \cdot r_i^\alpha|$
6. **Weight Renormalization**: $w_i \leftarrow w_i \cdot T / \sum_j w_j$

Where:
- $\bar{G} = \text{mean}(G_i)$ is the average gradient norm
- $\alpha$ is an asymmetry hyperparameter (higher = more aggressive balancing)
- $T$ is the number of tasks (sum of weights is preserved)

## Critical Implementation Details

### 1. Gradient Computation Order

The gradient norm must be computed as:
```
G_i = ||w_i * ∇_W L_i||
```

**NOT** as `||∇_W(w_i * L_i)||`. The difference:
- Correct: Compute gradient of **unweighted** loss first, then multiply by weight
- Wrong: Compute gradient of weighted loss directly

### 2. Weight Gradient Isolation

Weights are updated **ONLY** by the GradNorm loss, not by the main task loss:

```python
# After backward pass for task loss
total_loss.backward(retain_graph=True)

# Zero out weight gradients - they shouldn't be updated by task loss!
if self.weights.grad is not None:
    self.weights.grad.data.zero_()

# Then compute GradNorm loss and update weights separately
```

### 3. Weight Constraints

- Weights are clamped to positive values: $w_i \geq 0$
- Weights are renormalized so their sum equals $T$ (number of tasks)
- **Individual weights CAN exceed 1.0** - this is valid as long as they sum to $T$

## Configuration

In YAML config files:

```yaml
train:
  gradnorm:
    enabled: true
    alpha: 1.5        # Asymmetry parameter (1.0-2.0 typical)
    lr_weights: 0.025 # Learning rate for weight updates
    use_simple: false # true = gradient-free heuristic, false = full algorithm
```

## Two Implementations

1. **`GradNorm`** (Full): Paper-faithful implementation using actual gradient norms
2. **`GradNormSimple`** (Simplified): Gradient-free heuristic using loss ratios only

The full implementation is more principled but computationally heavier. The simplified version is easier to integrate but less effective.

## Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| PDE loss explodes | Weights updated by task loss gradient | Ensure weight gradients are zeroed after task backward |
| Weights all become equal | Alpha too low | Increase alpha (try 1.5-2.0) |
| Training unstable | lr_weights too high | Reduce lr_weights |
| No effect on balancing | GradNorm disabled or wrong shared layer | Check config and `get_shared_layer()` |

## References

- Original Paper: https://arxiv.org/abs/1711.02257
- Reference Implementation: https://github.com/brianlan/pytorch-grad-norm
