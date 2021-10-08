from pathlib import Path
import hydra
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import AdamW, AutoTokenizer, get_scheduler
from sklearn.model_selection import StratifiedKFold
from dataset import TextDataset
from models import (Attention_Pooling_Model, Conv_Pooling_Model,
                    Max_Pooling_Model, Mean_Max_Pooling_Model,
                    Mean_Pooling_Model, Transformer, Transformer_CLS_Embeddings,
                    Transformer_Pooler_Outputs)
from train import evaluate_fn, train_fn
from losses import bcewithlogits_loss_fn, crossentropy_loss_fn, mse_loss_fn



logger = logging.getLogger(__name__)


# initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_loader(
    train_data,
    train_data_text,
    train_data_label,
    pretrained_model_name,
    batch_size,
    val_data

):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)


    train_ds = TextDataset(
        train_data[train_data_text].values,
        train_data[train_data_label].values,
        tokenizer,
    )
   
    valid_ds = TextDataset(
            val_data[train_data_text].values, val_data[train_data_label].values, tokenizer
            )
            
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )       
    valid_dl = DataLoader(
            valid_ds, batch_size=batch_size*2, num_workers=4, pin_memory=True
            )
            
    return train_dl, valid_dl


#@torch.cuda.empty_cache()
@hydra.main(config_path="./configs", config_name="config")
def run_with_folds(cfg):

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.model_name}")
    run = wandb.init(
        project=cfg.wandb.project_name, group=cfg.wandb.group_name, save_code=True,config=cfg
    )


    data = pd.read_csv(cfg.dataset.train_data_path)
    data = data[[cfg.dataset.train_text, cfg.dataset.train_label]]
    data = data.sample(frac=1).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    # Do Kfolds training and cross validation
    kf = StratifiedKFold(n_splits=cfg.dataset.n_splits)

    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=data[cfg.dataset.train_text].values, y=data[cfg.dataset.train_label].values)):

        
        train_data = data.loc[train_idx]
        valid_data = data.loc[valid_idx]
        
        train_ds = TextDataset(
            train_data[cfg.dataset.train_text].values,
            train_data[cfg.dataset.train_label].values,
            tokenizer
            )
        train_dataloader = DataLoader(train_ds, cfg.training.batch_size,
                            shuffle=True, num_workers=4)
                            
        valid_ds =  TextDataset(
            valid_data[cfg.dataset.train_text].values,
            valid_data[cfg.dataset.train_label].values,
            tokenizer
            )

        val_dataloader = DataLoader(valid_ds, cfg.training.batch_size*2,
                            shuffle=True, num_workers=4)
        
          # set model architecture
        if cfg.model.model_type == "Attention_Pooling_Model":
            model = Attention_Pooling_Model(
                cfg.model.model_name, cfg.training.dropout, cfg.training.num_labels
                )
                
        elif cfg.model.model_type == "Mean_Pooling_Model":
            model = Mean_Pooling_Model(
            cfg.model.model_name, cfg.training.dropout, cfg.training.num_labels
        )
        
        elif cfg.model.model_type == "Transformer_Pooler_Outputs":
            model = Transformer_Pooler_Outputs(
            cfg.model.model_name, cfg.training.dropout, cfg.training.num_labels
        )
        
        elif cfg.model.model_type == "Transformer_CLS_Embeddings":
            model = Transformer_CLS_Embeddings(
            cfg.model.model_name, cfg.training.dropout, cfg.training.num_labels
        )
        
        elif cfg.model.model_type == "Max_Pooling_Model":
            model = Max_Pooling_Model(
            cfg.model.model_name, cfg.training.dropout, cfg.training.num_labels
        )
        
        elif cfg.model.model_type == "Mean_Max_Pooling_Model":
            model = Mean_Max_Pooling_Model(
            cfg.model.model_name, cfg.training.dropout, cfg.training.num_labels
        )
        
        elif cfg.model.model_type == "Conv_Pooling_Model":
            model = Conv_Pooling_Model(
            cfg.model.model_name, cfg.training.dropout, cfg.training.num_labels
        )
        
        else:
            model = Transformer(
            cfg.model.model_name, cfg.training.dropout, cfg.training.num_labels
        )
        
        logger.info(f"Using the model architecture: {cfg.model.model_type}")
        
        train_dataloader,  val_dataloder = get_loader(train_data, cfg.dataset.train_text, 
        cfg.dataset.train_label,cfg.model.model_name,cfg.training.batch_size, valid_data)
        model.to(device)
        
        no_decay = ["bias", "LayerNorm.Weight"]
        
        optimized_group_parameters = [
            {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ]
        },
        ]
        
        optimizer = AdamW(
            optimized_group_parameters,
            cfg.training.lr,
            weight_decay=cfg.training.weight_decay
            )
            
        nb_train_steps = int(
            len(train_dataloader) / cfg.training.batch_size * cfg.training.max_epochs
            )
            
        scheduler = get_scheduler(
            cfg.training.scheduler,
            optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=nb_train_steps
            )

        # set loss function
        logger.info(f"Using  {cfg.training.loss_func} as Loss function")
        if cfg.training.loss_func == "bceloss":
            loss_func = bcewithlogits_loss_fn

        elif cfg.training.loss_func == "crossentropy":
            loss_func = crossentropy_loss_fn

        else:
            loss_func = mse_loss_fn


        best_valid_loss = float("inf")

        train_losses = []
        train_accs = []

        valid_losses = []
        valid_accs = []

        logger.info("Start Training")

        for epoch in range(cfg.training.max_epochs):

            
            artifact = wandb.Artifact('model', type='model')

            train_loss, train_acc, lr = train_fn(
                train_dataloader,
                model,
                loss_func,
                optimizer,
                epoch,
                scheduler,
                cfg.training.batch_size,
            )
            valid_loss, valid_acc = evaluate_fn(
                    val_dataloader, model, loss_func, epoch, cfg.training.batch_size
                    )

            train_losses.extend(train_loss)
            train_accs.extend(train_acc)

            valid_losses.extend(valid_loss)
            valid_accs.extend(valid_acc)

            epoch_train_loss = np.mean(train_loss)
            epoch_train_acc = np.mean(train_acc)

            epoch_valid_loss = np.mean(valid_loss)
            epoch_valid_acc = np.mean(valid_acc)

            run.log(
                {
                    "Epoch Train Loss": epoch_train_loss,
                    "Epoch Train Accuracy": epoch_train_acc,
                    "Epoch valid Loss": epoch_valid_loss,
                    "Epoch Valid Accuracy": epoch_valid_acc,
                }
            )

                
            if epoch_valid_loss < best_valid_loss:
                    
                best_valid_loss = epoch_valid_loss

                logger.info(f"Saving best model in : {cfg.dataset.save_model_path}")
                    
                save_model_path = Path(cfg.dataset.save_model_path).stem
                save_model_path = f"{save_model_path}_{fold}_.bin"
                torch.save(model.state_dict(), save_model_path)
                artifact.add_file(save_model_path)
                

        
       
        logger.info(f"epoch: {epoch+1}")
        logger.info(
            f"train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}"
        )
        logger.info(
                f"valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}"
                )
                
                
                
if __name__ == "__main__":
  run_with_folds()
