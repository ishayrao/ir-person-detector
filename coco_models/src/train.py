#trainign script for all models except yolo
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import logging
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import generalized_box_iou_loss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.custom_detector import ThermalDetector
from models.faster_rcnn_detector import FasterRCNNDetector
from models.effnet_detector import EfficientNetDetector
from models.ssdlite_detector import SSDLiteDetector
from datasets.flir_dataset import FLIRDataset
from utils.transforms import build_transforms, custom_collate_fn
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, cfg: DictConfig):
    # Create tensorboard writer using Hydra's output directory
    writer = SummaryWriter(Path(cfg.logging.save_dir) / 'tensorboard')  # Use Hydra's output directory
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.training.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
    
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.training.num_epochs} [Train]')
        for batch_idx, (data, targets) in enumerate(train_pbar):
            # Move data to device
            data = data.to(device)
            # move target tensors to device as list of dicts 
            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets] #standard format for all models
            
            optimizer.zero_grad()

            if cfg.model.name in ["faster_rcnn", "ssdlite"]: #for torchvision models: they return a Dict[Tensor] which contains classification and regression losses
                loss_dict = model(data, targets_on_device)
                loss = sum(loss_dict.values()) #sum of classification and regression losses
            else:
                outputs = model(data) #standard format for custom pytorch models 
                loss_dict = compute_loss(outputs, targets_on_device)
                loss = loss_dict['total_loss']

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) # gradient clipping to avoid exploding gradients
            optimizer.step()
            train_loss += loss.item() #maintain running loss total 
            # Log individual losses per batch to tensorboard
            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar(f'BatchLoss/{loss_name}', loss_value.item(), 
                                epoch * len(train_loader) + batch_idx)

                # Update progress bar
            train_pbar.set_postfix({'loss this batch': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
    
        # Choose validation function based on model type
        if cfg.model.name in ["faster_rcnn", "ssdlite"]:
            avg_val_loss = evaluate_validation(model, val_loader, device, epoch, cfg)
        else:
            avg_val_loss = evaluate_validation_custom(model, val_loader, device, epoch, cfg)

        scheduler.step(avg_val_loss)

        # Log epoch metrics to tensorboard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        # Log learning rate
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log epoch metrics
        log.info(f'Epoch {epoch+1}/{cfg.training.num_epochs}: '
                f'Avg Train Loss: {avg_train_loss:.4f}, '
                f'Avg Val Loss: {avg_val_loss:.4f}')


        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(cfg.logging.save_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            patience_counter = 0
        else:
             patience_counter += 1
        
        #early stopping
        if patience_counter >= cfg.training.early_stopping_patience: #10 epochs without improvement 
            log.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

    writer.close()
    return best_val_loss  # for optuna to minimize the validation loss 

def evaluate_validation(model, val_loader, device, epoch, cfg: DictConfig):
    
    was_training = model.training #boolean flag to return model to mode it was in before evaluation

    # force loss-returning behavior
    model.train()

    # workaround to freeze batchnorm and dropout layers to aviod training them
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.eval()

    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Log validation batch info
            #log.info(f"Validation batch {batch_idx}")
            #log.info(f"Target boxes: {[t['boxes'].shape for t in targets]}")
        
            loss_dict = model(images, targets)
            
            batch_loss = sum(loss_dict.values()).item()
            val_loss += batch_loss
        

    # back to train mode 
    model.train(was_training)
    
    return val_loss / len(val_loader)

def evaluate_validation_custom(model, val_loader, device, epoch, cfg: DictConfig):
    model.eval()
    val_loss = 0.0
    criterion = compute_loss
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{cfg.training.num_epochs} [Val]')
        for i, (data, targets) in enumerate(val_pbar):
            data = data.to(device)
            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(data)
            loss = criterion(outputs, targets_on_device)
            val_loss += loss.item()
            val_pbar.set_postfix({'val_loss': loss.item()})

    return val_loss / len(val_loader)

def compute_loss(predictions, targets): #only for custom detector since torchvision has build in loss funcs 
    total_cls_loss = 0
    total_box_loss = 0
    batch_size = len(targets)
    
    for idx in range(batch_size):
        # Classification loss (binary since there is only one class)
        cls_loss = F.binary_cross_entropy_with_logits(
            predictions['scores'][idx], 
            targets[idx]['labels'].float()
        )
        
        # loss using GIoU for bounding boxes 
        box_loss = generalized_box_iou_loss(
            predictions['boxes'][idx],
            targets[idx]['boxes'],
            reduction='mean'
        )
        
        total_cls_loss += cls_loss
        total_box_loss += box_loss
    
        return {
        'classification_loss': total_cls_loss / batch_size,
        'box_loss': total_box_loss / batch_size,
        'total_loss': (total_cls_loss + total_box_loss) / batch_size
    }

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Log config
    log.info(f"check device: {torch.cuda.is_available()}")
    log.info(f"config: \n{OmegaConf.to_yaml(cfg)}")
    
    torch.manual_seed(cfg.experiment.seed)
    device = cfg.model.device
    log.info(f"device is: {device}")
    # model created based on config
    if cfg.model.name == "custom_detector":
        model = ThermalDetector(cfg).to(device)
        log.info("custom detector model created and moved to device")
    elif cfg.model.name == "faster_rcnn":
        model = FasterRCNNDetector(cfg).to(device)
        log.info("faster rcnn model created and moved to device")
    elif cfg.model.name == "effnet":
        model = EfficientNetDetector(cfg).to(device)
        log.info("efficientnet model created and moved to device")
    elif cfg.model.name == "ssdlite":
        model = SSDLiteDetector(cfg).to(device)
        log.info("ssdlite model created and moved to device")
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")
   
    #transforms 
   # if cfg.model.name in ["ssdlite", "faster_rcnn"]:
        train_transform = model.transforms # make these specific to each model 
        val_transform = model.transforms
   # else: #for custom and effnet 
    train_transform = build_transforms(cfg, is_train=True)
    val_transform = build_transforms(cfg, is_train=False)
    
    # Create datasets
    train_dataset = FLIRDataset(
        json_file=Path(cfg.dataset.data.train_annotations),
        thermal_dir=Path(cfg.dataset.data.train_images),
        transform=train_transform
        # can add device parameter here to avoid moving to device? 
    )
    
    val_dataset = FLIRDataset(
        json_file=Path(cfg.dataset.data.val_annotations),
        thermal_dir=Path(cfg.dataset.data.val_images),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=custom_collate_fn 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=custom_collate_fn
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )

    scheduler = ReduceLROnPlateau(optimizer, 
                             mode='min',           # minimize validation loss
                             factor=0.1,           # reduce LR by factor of 10
                             patience=5,              
                             min_lr=1e-6)  
    
    # train model and get best validation loss
    best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg
    )
    
    # best validation loss for Optuna 
    return best_val_loss

if __name__ == "__main__":
    main() 