import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from ultralytics import YOLO
import torch
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = 'cpu'
    print("No GPU available, using CPU")

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup logging
    log.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    torch.manual_seed(cfg.experiment.seed)
    
    # Initialize YOLO model
    model = YOLO(f"yolo{cfg.model.model.version}.pt")
    
    # Train the model
    results = model.train(
        data=cfg.dataset.data.train,  # Path to training data from dataset config
        epochs=cfg.model.model.epochs,
        imgsz=cfg.model.model.img_size,
        batch=cfg.model.model.batch_size,
        device=cfg.model.model.device,
        workers=cfg.training.num_workers,
        project=cfg.logging.save_dir,
        name=f"yolo_{cfg.model.model.version}",
        exist_ok=True,
        pretrained=cfg.model.model.pretrained,
        optimizer=cfg.model.training.optimizer,
        lr0=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay,
        momentum=cfg.model.training.momentum,
        warmup_epochs=cfg.model.training.warmup_epochs,
        warmup_momentum=cfg.model.training.warmup_momentum,
        warmup_bias_lr=cfg.model.training.warmup_bias_lr,
        box=cfg.model.training.box,
        cls=cfg.model.training.cls,
        dfl=cfg.model.training.dfl,
        pose=cfg.model.training.pose,
        kobj=cfg.model.training.kobj,
        label_smoothing=cfg.model.training.label_smoothing,
        nbs=cfg.model.training.nbs,
        overlap_mask=cfg.model.training.overlap_mask,
        mask_ratio=cfg.model.training.mask_ratio,
        dropout=cfg.model.training.dropout,
        val=cfg.model.training.val,
        plots=cfg.model.training.plots
    )
    
    # Save the final model
    model.save(f"{cfg.logging.save_dir}/yolo_{cfg.model.model.version}/weights/best.pt")

if __name__ == "__main__":
    main()
