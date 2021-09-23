
import json
import logging

import hydra
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf.omegaconf import OmegaConf
from transformers import AutoTokenizer

from dataset import TextDataset

from models import (Attention_Pooling_Model, Conv_Pooling_Model,
                    Max_Pooling_Model, Mean_Max_Pooling_Model,
                    Mean_Pooling_Model, Transformer, Transformer_CLS_Embeddings,
                    Transformer_Pooler_Outputs)


logger = logging.getLogger(__name__)

def create_dl(data_path, model_name, text_column, label_column):
    data = pd.read_csv(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = TextDataset(
        data[text_column].values, data[label_column].values, tokenizer, train=False
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

    return dataloader


def get_predictions(dataloader, model_type, model_path, model_name, dropout, num_labels, device):

    # set model architecture
    if model_type == "Attention_Pooling_Model":
        model = Attention_Pooling_Model(
            model_name, dropout, num_labels
        )

    elif model_type == "Mean_Pooling_Model":
        model = Mean_Pooling_Model(
            model_name, dropout, num_labels
        )

    elif model_type == "Transformer_Pooler_Outputs":
        model = Transformer_Pooler_Outputs(
           model_name, dropout, num_labels
        )

    elif model_type == "Transformer_CLS_Embeddings":
        model = Transformer_CLS_Embeddings(
            model_name, dropout, num_labels
        )

    elif model_type == "Max_Pooling_Model":
        model = Max_Pooling_Model(
            model_name, dropout, num_labels
        )

    elif model_type == "Mean_Max_Pooling_Model":
        model = Mean_Max_Pooling_Model(
            model_name, dropout, num_labels
        )

    elif model_type == "Conv_Pooling_Model":
        model = Conv_Pooling_Model(
            model_name, dropout, num_labels
        )

    else:
        model = Transformer(
            model_name, dropout, num_labels
        )


    model.load_state_dict(torch.load(model_path,map_location=device))
    model.to(device)

    preds = []
    for idx, inputs in enumerate(tqdm(dataloader)):

        input_ids = inputs["input_ids"].to(device, dtype=torch.long)
        attention_mask = inputs["attention_mask"].to(device, dtype=torch.long)
        outputs = model(input_ids, attention_mask)
        preds.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return preds


def compute_metrics(data_path, preds, text_col_name,label_column_name, metrics_save_path, save_preds_path):

    outputs = np.array(preds) >= 0.5
    data = pd.read_csv(data_path)
    targets = data[label_column_name].values
    weighted_f1_score = f1_score(targets, outputs, average="weighted")
    micro_f1_score = f1_score(targets, outputs, average="micro")
    macro_f1_score = f1_score(targets, outputs, average="macro")
    accuracy = accuracy_score(targets, outputs)
    mcc = matthews_corrcoef(targets, outputs)

    metrics = {
        "Weighted F1 score": weighted_f1_score,
        "Micro f1 score": micro_f1_score,
        "Macro f1 score": macro_f1_score,
        "Accuracy": accuracy,
        "MCC": mcc,
    }

    data = pd.DataFrame({'Text': data[text_col_name].values, 'Actual Value': data[label_column_name].values,
                          'Predicted Value': outputs.flatten() })
    data.to_csv(save_preds_path)

    with open(metrics_save_path, "w") as file:
        json.dump(metrics, file)


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    dataloader = create_dl(
        cfg.dataset.test_data_path, cfg.model.model_name, 
        cfg.dataset.test_text, cfg.dataset.test_label
    )
    dropout=0
    preds = get_predictions(
        dataloader,
        cfg.model.model_type,
        cfg.dataset.finetuned_model_path,
        cfg.model.model_name,
        dropout,
        cfg.training.num_labels,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    compute_metrics(cfg.dataset.test_data_path, preds, cfg.dataset.test_text, cfg.dataset.test_label, cfg.dataset.save_metrics_path,
                      cfg.dataset.save_preds_path)


if __name__ == "__main__":
    main()