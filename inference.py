
import json

import hydra
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import TangDataset
from models import BERT, BertAttnhead, MeanPoolingModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained models")
    # data path
    parser.add_argument(
        "--data_path", type=str, default=None, help="The path to the test dataset"
    )
    # text columns
    parser.add_argument(
        "--text_column", type=str, default=None, help="Text column name"
    )
    # label columns
    parser.add_argument(
        "--label_column", type=str, default=None, help="Label column name"
    )
    # pretrained model name
    parser.add_argument(
        "--pretrained_model_name", type=str, default=None, help="Pretrained model name"
    )
    # model path
    parser.add_argument(
        "--model_path", type=str, default=None, help="The path to the pretrained model"
    )
    # model type
    parser.add_argument(
        "--model_type", type=str, default=None, help="Type of the model"
    )
    # metrics file path
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=None,
        help="The path to save the metrics file",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.data_path is None:
        raise ValueError("Need a dataset for testing")

    else:
        if args.data_path is not None:
            extension = args.data_path.split(".")[-1]
            assert extension in "csv", "`data file` should be a csv file."

    return args


def create_dl(data_path, model_name, text_column, label_column):
    data = pd.read_csv(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = TangDataset(
        data[text_column].values, data[label_column].values, tokenizer, train=False
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

    return dataloader


def get_predictions(dataloader, model_type, model_path, model_name, device):

    if model_type == "MeanPoolingModel":
        model = MeanPoolingModel(model_name)
    elif model_type == "AttentionHead":
        model = BertAttnhead(model_name)
    else:
        model = BERT(model_name)

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    preds = []
    for idx, inputs in enumerate(tqdm(dataloader)):

        input_ids = inputs["input_ids"].to(device, dtype=torch.long)
        attention_mask = inputs["attention_mask"].to(device, dtype=torch.long)
        outputs = model(input_ids, attention_mask)
        preds.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return preds


def compute_metrics(data_path, preds, label_column_name, metrics_save_path):

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

    with open(metrics_save_path, "w") as file:
        json.dump(metrics, file)

@
def main():
    args = parse_args()
    dataloader = create_dl(
        args.data_path, args.pretrained_model_name, args.text_column, args.label_column
    )
    preds = get_predictions(
        dataloader,
        args.model_type,
        args.model_path,
        args.pretrained_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    compute_metrics(args.data_path, preds, args.label_column, args.metrics_path)


if __name__ == "__main__":
    main()