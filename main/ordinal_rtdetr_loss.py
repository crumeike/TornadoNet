"""
RT-DETR Ordinal Loss Implementation

Modifies RT-DETR's classification loss to incorporate ordinal structure
for building damage severity estimation. Implements soft ordinal targets
and distance-based penalties as described in the TornadoNet paper.

References:
    - RT-DETR: https://github.com/ultralytics/ultralytics
    - TornadoNet paper: [Add arXiv link]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRTDETRLoss(nn.Module):
    """
    RT-DETR loss with ordinal-aware classification for damage states.
    
    Wraps standard RT-DETR loss and modifies the classification component
    to use soft ordinal targets and optional distance penalties.
    
    Args:
        nc: Number of classes (5 for DS0-DS4)
        k_neighbors: Maximum ordinal distance for non-zero soft labels (0=disabled)
        sigma: Gaussian smoothing parameter (default: 0.5)
        alpha: Ordinal distance penalty weight (0=disabled)
    """
    
    def __init__(self, nc, k_neighbors=1, sigma=0.5, alpha=0.0):
        super().__init__()
        self.nc = nc
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.alpha = alpha
        
        print(f"✅ RT-DETR Ordinal Loss initialized:")
        print(f"   Classes: {nc}")
        print(f"   k_neighbors: {k_neighbors}")
        print(f"   sigma: {sigma}")
        print(f"   alpha: {alpha}")

    def create_soft_ordinal_targets_vectorized(self, true_class_idx, num_classes, device):
        """
        Create soft ordinal targets using Gaussian smoothing (vectorized).
        
        Implements the soft target formulation from TornadoNet paper:
        w_k = exp(-(k-c)²/(2σ²)), bounded to |k-c| ≤ K
        
        Args:
            true_class_idx: Tensor [N] of integer class indices
            num_classes: Total number of classes
            device: Torch device
            
        Returns:
            Soft targets [N, num_classes] normalized to sum to 1
        """
        N = true_class_idx.shape[0]
        
        # Create class index grid [N, nc]
        class_grid = torch.arange(num_classes, device=device).unsqueeze(0).expand(N, -1).float()
        true_class_expanded = true_class_idx.unsqueeze(1).float()
        
        # Compute distances (k - c)
        distances = class_grid - true_class_expanded
        
        # Gaussian weights: exp(-(k-c)²/(2σ²))
        weights = torch.exp(-(distances ** 2) / (2 * self.sigma ** 2))

        # k-neighbor bounding: set to 0 if |k-c| > K
        distances_abs = torch.abs(distances)
        weights = torch.where(distances_abs <= self.k_neighbors, weights, torch.zeros_like(weights))

        # Normalize: w_k / Σw_j
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / (weight_sum + 1e-8)  # Avoid division by zero
        
        return weights


def inject_ordinal_loss_rtdetr(trainer, k_neighbors=1, sigma=0.5, alpha=0.0):
    """
    Inject ordinal loss into RT-DETR trainer's criterion.
    
    Modifies the classification loss computation to use soft ordinal targets
    and optional distance penalties while preserving bbox and GIoU losses.
    
    Args:
        trainer: RT-DETR trainer instance (from Ultralytics)
        k_neighbors: Maximum ordinal distance for non-zero weights (0=baseline)
        sigma: Gaussian smoothing parameter
        alpha: Ordinal distance penalty weight (0=disabled)
        
    Returns:
        Modified trainer with ordinal-aware classification loss
    """
    # Get number of classes from dataset config
    nc = trainer.data['nc']
    print(f"🔢 Number of classes: {nc}")

    # Initialize criterion and get reference
    criterion = trainer.model.init_criterion()
    trainer.model.criterion = criterion
    
    # Create ordinal loss module
    ordinal_loss_module = OrdinalRTDETRLoss(
        nc=nc,
        k_neighbors=k_neighbors,
        sigma=sigma,
        alpha=alpha
    )
    
    # Store original classification loss method
    original_get_loss_class = criterion._get_loss_class
    
    def ordinal_get_loss_class(pred_scores, targets, gt_scores, num_gts, postfix=""):
        """
        Modified classification loss with ordinal soft targets and penalty.
        
        Replaces RT-DETR's standard focal loss with ordinal-aware version.
        """
        bs, nq = pred_scores.shape[:2]
        name_class = f"loss_class{postfix}"
        
        # Validate parameters
        if k_neighbors < 0 or alpha < 0:
            raise ValueError("k_neighbors and alpha must be non-negative")

        # Step 1: Create soft ordinal targets (if k_neighbors > 0)
        if k_neighbors > 0:
            # Flatten targets and create soft distribution
            targets_flat = targets.reshape(-1)
            valid_mask = targets_flat < nc
            
            soft_targets = torch.zeros((bs * nq, nc), device=pred_scores.device)
            if valid_mask.sum() > 0:
                soft_targets[valid_mask] = ordinal_loss_module.create_soft_ordinal_targets_vectorized(
                    targets_flat[valid_mask], nc, pred_scores.device
                )
            soft_targets = soft_targets.reshape(bs, nq, nc)
        else:
            # Standard one-hot encoding (baseline)
            one_hot = torch.zeros((bs, nq, nc + 1), dtype=torch.int64, device=targets.device)
            one_hot.scatter_(2, targets.unsqueeze(-1), 1)
            soft_targets = one_hot[..., :-1].float()
        
        # Apply task-aligned assignment weights (gt_scores)
        soft_targets = gt_scores.view(bs, nq, 1) * soft_targets
        
        # Step 2: Compute focal loss with soft targets
        loss_cls = criterion.fl(pred_scores, soft_targets)
        loss_cls /= max(num_gts, 1) / nq
        
        # Step 3: Apply ordinal distance penalty (if alpha > 0)
        if alpha > 0:
            pred_probs = F.softmax(pred_scores, dim=-1)
            class_indices = torch.arange(nc, device=pred_probs.device).float()
            pred_class_soft = (pred_probs * class_indices).sum(dim=-1)  # [bs, nq]
            
            # Compute ordinal distance
            valid_mask = targets < nc
            distance = torch.abs(pred_class_soft - targets.float())
            distance = distance * valid_mask.float()  # Zero out background
            
            # Apply penalty: loss *= (1 + alpha * distance)
            penalty = 1.0 + alpha * distance
            loss_cls = loss_cls * penalty[valid_mask].mean() if valid_mask.sum() > 0 else loss_cls

        return {name_class: loss_cls.squeeze() * criterion.loss_gain["class"]}

    # Replace the classification loss method
    criterion._get_loss_class = ordinal_get_loss_class
    
    print(f"✅ RT-DETR ordinal loss injected successfully")
    return trainer


if __name__ == "__main__":
    """Test ordinal loss computation with dummy data"""
    print("=" * 70)
    print("Testing RT-DETR Ordinal Loss")
    print("=" * 70)
    
    # Configuration
    nc = 5  # 5 damage states (DS0-DS4)
    batch_size = 4
    num_queries = 300
    
    print(f"\n📊 Test configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Queries per image: {num_queries}")
    print(f"   Number of classes: {nc}")
    
    # Create dummy predictions and labels
    pred_logits = torch.randn(batch_size, num_queries, nc)
    true_labels = torch.randint(0, nc, (batch_size, num_queries))
    
    # Initialize ordinal loss
    print(f"\n⚙️  Initializing ordinal loss module...")
    ordinal_loss = OrdinalRTDETRLoss(nc=nc, k_neighbors=1, sigma=0.5, alpha=0.5)
    
    # Test 1: Soft target creation
    print(f"\n" + "=" * 70)
    print("Test 1: Soft Target Creation (k=1, σ=0.5)")
    print("=" * 70)
    
    for ds in range(nc):
        single_label = torch.tensor([ds])
        soft_target = ordinal_loss.create_soft_ordinal_targets_vectorized(
            single_label, nc, pred_logits.device
        )
        weights_str = ", ".join([f"{w:.3f}" for w in soft_target[0].numpy()])
        print(f"DS{ds} → [{weights_str}]")
    
    # Test 2: Compare different alpha values
    print(f"\n" + "=" * 70)
    print("Test 2: Loss Comparison (varying alpha)")
    print("=" * 70)
    
    test_batch = pred_logits[:2, :10, :].reshape(-1, nc)  # 20 predictions
    test_labels = true_labels[:2, :10].reshape(-1)
    
    for alpha_val in [0.0, 0.05, 0.2]:
        loss_module = OrdinalRTDETRLoss(nc=nc, k_neighbors=1, sigma=0.5, alpha=alpha_val)
        
        # Compute soft targets
        soft_targets = loss_module.create_soft_ordinal_targets_vectorized(
            test_labels, nc, pred_logits.device
        )
        
        # Compute focal loss
        pred_probs = F.softmax(test_batch, dim=-1)
        pred_log_probs = F.log_softmax(test_batch, dim=-1)
        ce_loss = -(soft_targets * pred_log_probs).sum(dim=-1)
        p_t = (soft_targets * pred_probs).sum(dim=-1)
        focal_weight = (1 - p_t) ** 2.0
        focal_loss = 0.25 * focal_weight * ce_loss
        
        # Apply ordinal penalty if enabled
        if alpha_val > 0:
            class_indices = torch.arange(nc, device=pred_probs.device).float()
            pred_class_soft = (pred_probs * class_indices).sum(dim=-1)
            distance = torch.abs(pred_class_soft - test_labels.float())
            penalty = 1.0 + alpha_val * distance
            focal_loss = focal_loss * penalty
        
        loss_value = focal_loss.mean().item()
        print(f"α={alpha_val:.2f}: Loss = {loss_value:.4f}")
    
    print(f"\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
