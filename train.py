"""
TornadoNet Training Script
Supports both standard and ordinal-aware loss training for YOLO and RT-DETR models
"""

import torch
from ultralytics import YOLO, RTDETR
from ultralytics.utils.loss import v8DetectionLoss
from ordinal_rtdetr_loss import inject_ordinal_loss_rtdetr


class OrdinalDetectionLoss(v8DetectionLoss):
    """
    YOLO Detection Loss with k-Neighbor Ordinal Supervision
    
    Applies Gaussian-smoothed labels constrained to k nearest ordinal neighbors,
    preventing reward for extreme classification errors.
    
    Args:
        model: YOLO model instance
        sigma: Gaussian smoothing parameter (default: 0.5)
        k_neighbors: Maximum ordinal distance for non-zero probability (default: 0)
                     k=0: Standard BCE loss (baseline)
                     k=1: Only ±1 neighbor gets non-zero weight (recommended)
                     k=2: Only ±2 neighbors (more permissive)
        alpha: Ordinal distance penalty weight (default: 0.0)
               alpha=0.0: No ordinal penalty
               alpha>0.0: Penalize based on predicted vs true class distance
    """
    
    def __init__(self, model, sigma=0.5, k_neighbors=0, alpha=0.0):
        super().__init__(model)
        self.sigma = sigma
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        
        # Print configuration
        if self.k_neighbors <= 0 and self.alpha == 0:
            print("✅ Using standard BCE Loss (baseline)")
        else:
            if self.k_neighbors > 0:
                print(f"🎯 Ordinal bounding enabled: k={k_neighbors}")
            if self.alpha > 0:
                print(f"🎯 Ordinal distance penalty enabled: α={alpha}")
            print(f"📊 Ordinal Loss config: σ={sigma}, k={k_neighbors}, α={alpha}")

    def create_soft_ordinal_targets(self, target_scores, gt_labels, target_gt_idx, fg_mask, batch_size):
        """
        Convert one-hot targets to k-neighbor bounded ordinal distributions.
        
        Example with k=1, true_class=DS2, σ=0.5:
            Full Gaussian:  [0.05, 0.24, 0.40, 0.24, 0.05]
            k=1 Bounded:    [0.00, 0.38, 0.62, 0.00, 0.00]  ← Only DS1, DS2, DS3
        
        Returns:
            Soft target distribution scaled by task-aligned assignment strength
        """
        soft_targets = target_scores.clone()
        num_classes = soft_targets.shape[-1]
        
        for b in range(batch_size):
            img_fg_mask = fg_mask[b]
            if img_fg_mask.sum() == 0:
                continue
            
            # Get ground truth labels for this image
            img_gt_labels = gt_labels[b].squeeze(-1)
            img_target_idx = target_gt_idx[b][img_fg_mask].long()
            img_target_idx = img_target_idx.clamp(0, len(img_gt_labels) - 1)
            true_classes = img_gt_labels[img_target_idx]
            
            # Process each foreground anchor
            for i, true_class in enumerate(true_classes):
                if true_class < 0:
                    continue
                
                # Calculate Gaussian weights for all classes
                class_indices = torch.arange(num_classes, device=soft_targets.device)
                distances = torch.abs(class_indices - true_class)
                weights = torch.exp(-(distances ** 2) / (2 * self.sigma ** 2))
                
                # Apply k-neighbor bounding: zero out classes beyond k neighbors
                weights[distances > self.k_neighbors] = 0.0
                
                # Normalize to sum to 1
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    # Fallback to one-hot if all weights zeroed
                    weights = torch.zeros_like(weights)
                    weights[int(true_class)] = 1.0
                
                # Scale by task-aligned assignment strength
                anchor_idx = torch.where(img_fg_mask)[0][i]
                strength = target_scores[b, anchor_idx].sum()
                soft_targets[b, anchor_idx] = weights * strength
        
        return soft_targets
    
    def __call__(self, preds, batch):
        """Calculate loss with optional k-neighbor bounded ordinal targets"""
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        # Split predictions into distribution and scores
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        
        # Create anchors
        from ultralytics.utils.tal import make_anchors
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # Prepare targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        # Task-aligned assignment
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum_box = max(target_scores.sum(), 1)

        # ============================================================================
        # CLASSIFICATION LOSS (with optional ordinal supervision)
        # ============================================================================
        if self.k_neighbors <= 0:
            # Standard BCE Loss (baseline)
            target_scores_sum_cls = max(target_scores.sum(), 1)
            bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        else:
            # k-Neighbor Bounded Ordinal Loss
            soft_target_scores = self.create_soft_ordinal_targets(
                target_scores, gt_labels, target_gt_idx, fg_mask, batch_size
            )
            target_scores_sum_cls = max(soft_target_scores.sum(), 1)
            bce_loss = self.bce(pred_scores, soft_target_scores.to(dtype))

        # Apply ordinal distance penalty if enabled
        if self.alpha > 0 and fg_mask.sum().item() > 0:
            ordinal_penalty = torch.ones_like(bce_loss)
            
            for b in range(batch_size):
                img_fg_mask = fg_mask[b]
                if img_fg_mask.sum() == 0:
                    continue
                    
                # Get true classes for foreground anchors
                img_gt_labels = gt_labels[b].squeeze(-1)
                img_target_idx = target_gt_idx[b][img_fg_mask].long().clamp(
                    0, len(img_gt_labels) - 1
                )
                true_classes = img_gt_labels[img_target_idx]
                
                # Get predicted classes (expected value using sigmoid probabilities)
                fg_preds = pred_scores[b, img_fg_mask].sigmoid()
                pred_classes_soft = (
                    fg_preds * torch.arange(self.nc, device=fg_preds.device)
                ).sum(dim=1)
                
                # Calculate ordinal distance penalty for each foreground anchor
                for i, (true_cls, pred_cls_soft) in enumerate(zip(true_classes, pred_classes_soft)):
                    if true_cls < 0:
                        continue
                    
                    distance = torch.abs(pred_cls_soft - true_cls)
                    anchor_idx = torch.where(img_fg_mask)[0][i]
                    
                    # Apply penalty: loss *= (1 + α * distance)
                    ordinal_penalty[b, anchor_idx] *= (1.0 + self.alpha * distance)
            
            loss[1] = (bce_loss * ordinal_penalty).sum() / target_scores_sum_cls
        else:  
            loss[1] = bce_loss.sum() / target_scores_sum_cls

        # ============================================================================
        # BBOX and DFL LOSSES (standard YOLO losses)
        # ============================================================================
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, 
                target_scores, target_scores_sum_box, fg_mask
            )
        
        loss[0] *= self.hyp.box 
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        
        return loss.sum() * batch_size, loss.detach()


def train_tornadonet(model_name='yolo11n.pt', data='data.yaml', ordinal_loss=False, 
                     sigma=0.5, k_neighbors=0, alpha=0.0, **kwargs):
    """
    Train TornadoNet models with optional ordinal supervision
    
    Args:
        model_name: Model checkpoint (e.g., 'yolo11n.pt', 'rtdetr-l.pt')
        data: Path to data.yaml
        ordinal_loss: Enable ordinal supervision (legacy flag, use k_neighbors or alpha instead)
        sigma: Gaussian smoothing parameter (default: 0.5, range: 0.1-2.0)
        k_neighbors: Maximum ordinal distance for non-zero weights
                     0: Standard loss (baseline)
                     1: Only ±1 neighbor (recommended for ordinal)
                     2: Only ±2 neighbors (more permissive)
        alpha: Ordinal distance penalty weight (default: 0.0)
               0.0: No penalty
               0.05-0.2: Mild to moderate penalty
        **kwargs: Additional training arguments passed to model.train()
    
    Returns:
        Training results object
    """
    # Load model
    print(f"\n📦 Loading model: {model_name}")
    if 'yolo' in model_name.lower():
        model = YOLO(model_name)
    elif 'rtdetr' in model_name.lower():
        model = RTDETR(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported: YOLO, RT-DETR")

    # Inject ordinal loss if requested
    def inject_yolo_ordinal_loss(trainer):
        """Callback to replace YOLO loss with ordinal variant"""
        print(f"\n⚙️  Injecting ordinal loss into YOLO criterion...")
        trainer.model.criterion = OrdinalDetectionLoss(
            trainer.model, 
            sigma=sigma, 
            k_neighbors=k_neighbors,
            alpha=alpha
        )

    def inject_rtdetr_ordinal_loss(trainer):
        """Callback to replace RT-DETR loss with ordinal variant"""
        print(f"\n⚙️  Injecting ordinal loss into RT-DETR criterion...")
        inject_ordinal_loss_rtdetr(
            trainer,
            k_neighbors=k_neighbors,
            sigma=sigma,
            alpha=alpha
        )
    
    # Register callbacks if ordinal loss is enabled
    use_ordinal = ordinal_loss or alpha > 0.0 or k_neighbors > 0
    
    if use_ordinal and 'yolo' in model_name.lower():
        print("🎯 Enabling ordinal supervision for YOLO...")
        model.add_callback('on_train_start', inject_yolo_ordinal_loss)
    elif use_ordinal and 'rtdetr' in model_name.lower():
        print("🎯 Enabling ordinal supervision for RT-DETR...")
        model.add_callback('on_train_start', inject_rtdetr_ordinal_loss)

    # Start training
    print(f"\n🚀 Starting training...")
    results = model.train(
        data=data,
        device=[0, 1] if torch.cuda.device_count() >= 2 else 0,
        **kwargs
    )
    
    return results


if __name__ == "__main__":
    import argparse
    import yaml
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description='TornadoNet Training Script - Train YOLO/RT-DETR with ordinal supervision'
    )
    
    # Model and data
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                        help='Model checkpoint (e.g., yolo11n.pt, rtdetr-l.pt)')
    parser.add_argument('--data', type=str, default='data/tornadonet.yaml', 
                        help='Path to data.yaml')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=896, help='Image size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Ordinal loss parameters
    parser.add_argument('--ordinal_loss', action='store_true', 
                        help='Enable ordinal loss (legacy flag)')
    parser.add_argument('--k_neighbors', type=int, default=0, 
                        help='k-neighbor bounding (0=baseline, 1=recommended, 2=permissive)')
    parser.add_argument('--sigma', type=float, default=0.5, 
                        help='Gaussian smoothing parameter (0.1-2.0)')
    parser.add_argument('--alpha', type=float, default=0.0, 
                        help='Ordinal distance penalty weight (0.0=disabled)')
    
    # Data augmentation
    parser.add_argument('--rect', action='store_true', help='Rectangular training')
    parser.add_argument('--multi_scale', action='store_true', help='Multi-scale training')
    parser.add_argument('--close_mosaic', type=int, default=10, 
                        help='Close mosaic augmentation in last N epochs')
    parser.add_argument('--augment', action='store_true', help='Test-time augmentation')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/tornadonet', 
                        help='Project directory')
    parser.add_argument('--name', type=str, default='train', 
                        help='Experiment name')
    parser.add_argument('--deterministic', action='store_true', 
                        help='Deterministic training for reproducibility')

    args = parser.parse_args()

    # Auto-disable deterministic for RT-DETR (unstable with deterministic=True)
    if 'rtdetr' in args.model.lower():
        args.deterministic = False
        print("⚠️  Deterministic training disabled for RT-DETR (known instability)")

    # Prepare experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = args.model.split('.')[0]
    
    use_ordinal = args.ordinal_loss or args.alpha > 0.0 or args.k_neighbors > 0
    if use_ordinal:
        if args.k_neighbors > 0:
            exp_name = f"{timestamp}_ordinal_k{args.k_neighbors}"
        elif args.alpha > 0.0:
            exp_name = f"{timestamp}_ordinal_alpha{args.alpha}"
        else:
            exp_name = f"{timestamp}_ordinal"
    else:
        exp_name = f"{timestamp}_baseline"
    
    args.name = f"{model_name}/{exp_name}"

    # Save configuration
    config_dir = f"{args.project}/{args.name}"
    os.makedirs(config_dir, exist_ok=True)
    
    config = vars(args)
    with open(f"{config_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n📝 Configuration saved to: {config_dir}/config.yaml")

    # Training arguments
    train_kwargs = {
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'patience': args.patience,
        'project': args.project,
        'name': args.name,
        'seed': args.seed,
        'deterministic': args.deterministic,
        'rect': args.rect,
        'multi_scale': args.multi_scale,
        'close_mosaic': args.close_mosaic,
        'augment': args.augment,
        'cache': True,
        'plots': True,
        'verbose': True,
        'save': True,
    }

    # Train model
    results = train_tornadonet(
        model_name=args.model,
        data=args.data,
        ordinal_loss=args.ordinal_loss,
        alpha=args.alpha,
        sigma=args.sigma,
        k_neighbors=args.k_neighbors,
        **train_kwargs
    )

    print(f"\n✅ Training completed!")
    print(f"📊 Results saved to: {results.save_dir}")
