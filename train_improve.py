import datetime
import os
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.segformer_rda import SegFormer
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent
    #---------------------------------#
                
    #---------------------------------#
    Cuda            = torch.cuda.is_available()
    #---------------------------------------------------------------------#
                                         
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
                                             
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
                                  
    #---------------------------------------------------------------------#
    fp16            = False
    #-----------------------------------------------------#
                                     
    #-----------------------------------------------------#
    num_classes     = 2
    #-------------------------------------------------------------------#
                                           
    #-------------------------------------------------------------------#
    phi             = 'b2'
    #-------------------------------------------------------------------#
                                      
    #-------------------------------------------------------------------#
    pretrained      = False
    #-------------------------------------------------------------------#
                               
    #-------------------------------------------------------------------#
    # Leave this empty for a fresh run. Set SEGFORMER_RDA_INIT_WEIGHTS to
    # resume from a SegFormer-RDA checkpoint or another compatible checkpoint.
    model_path      = os.environ.get("SEGFORMER_RDA_INIT_WEIGHTS", "")

    #------------------------------#
               
    #------------------------------#
    input_shape     = [512, 512]
    
    #----------------------------------------------------------------------------------------------------------------------------#
                                
    #----------------------------------------------------------------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 35
    Freeze_batch_size   = 16
    UnFreeze_Epoch      = 105
    Unfreeze_batch_size = 8

    Freeze_Train        = True

              
    Init_lr             = 5e-5
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adamw" 
    momentum            = 0.9
    weight_decay        = 3e-2           
    
    lr_decay_type       = 'cos'
    save_period         = 5
    save_dir            = os.environ.get("SEGFORMER_RDA_LOG_DIR", str(PROJECT_ROOT / "logs_improve"))
    eval_flag           = True
    eval_period         = 5
    
    #------------------------------#
             
    #------------------------------#
    VOCdevkit_path  = os.environ.get(
        "SEGFORMER_RDA_DATASET_ROOT",
        str(PROJECT_ROOT / "liefeng"),
    )
    
    dice_loss       = True
    focal_loss      = True
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers     = int(os.environ.get("SEGFORMER_RDA_NUM_WORKERS", "4"))

    seed_everything(11)
    
          
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
    else:
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank  = 0

    if local_rank == 0:
                                  
        show_config(num_classes=num_classes, phi=phi, pretrained=pretrained, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, Freeze_batch_size=Freeze_batch_size, \
            Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, \
            momentum=momentum, lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir, dice_loss=dice_loss, focal_loss=focal_loss)

           
    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained)

    if not pretrained:
        weights_init(model)
    if model_path != '':
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Initial weights were requested but not found: {model_path}. "
                "Set SEGFORMER_RDA_INIT_WEIGHTS to a valid checkpoint or leave it unset."
            )
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if sync_bn and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

           
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if True:
                
        batch_size = Freeze_batch_size if not distributed else Freeze_batch_size // ngpus_per_node
        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type in ['adam', 'adamw'] else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit     = max(batch_size * Init_lr / nbs, lr_limit_min)
        Min_lr_fit      = max(batch_size * Min_lr / nbs, lr_limit_min * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay = weight_decay),
            'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,) if distributed else None
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,) if distributed else None

        gen             = DataLoader(train_dataset, shuffle=(train_sampler is None), batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=seg_dataset_collate, sampler=val_sampler)

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

              
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and Freeze_Train:
                batch_size = Unfreeze_batch_size if not distributed else Unfreeze_batch_size // ngpus_per_node
                            
                Init_lr_fit = max(batch_size * Init_lr / nbs, lr_limit_min)
                Min_lr_fit  = max(batch_size * Min_lr / nbs, lr_limit_min * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                gen             = DataLoader(train_dataset, shuffle=(train_sampler is None), batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=seg_dataset_collate, sampler=val_sampler)
                Freeze_Train = False

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                          len(gen), len(gen_val), gen, gen_val, UnFreeze_Epoch, Cuda, 
                          dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
