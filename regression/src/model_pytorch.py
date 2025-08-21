from torch import nn, concat
from torch.optim import AdamW, lr_scheduler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import os
import json
import pandas as pd
import torch
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
import numpy as np
import iFeatureOmegaCLI
from transformers import T5Tokenizer, T5EncoderModel

from torch import mean

FEATURE_SHAPE = {
    "pubchem": 881,
    "mol2vec": 300,
    "ecfp4": 2048,
    "fcfp6": 2048,
    "extended": 1024,
    "scibert": 769,
    "molt5_small": 513,
    "ecfp6": 2048,
    "molt5_base": 769,
    "fp2": 1024,
    "klekota_roth": 4860,
    "avalon": 512,
    "molt5_large": 1025,
    "selformer": 769,
    "hybridization": 1024,
    "fcfp4": 2048,
    "chemberta": 769,
    "rdkit": 2048,
    "standard": 1024,
    "maccs": 167,
    "estate": 245,
}
PRETRAIN_MAPPING = {
    "molt5_base": "laituan245/molt5-base",
    "molt5_small": "laituan245/molt5-small",
    "molt5_large": "laituan245/molt5-large",
    "estate": "EState",
}
ADDUCT = {
    "[M-H+HCOOH]-": torch.tensor([1.0, 0.0]),
    "[M-H]-": torch.tensor([0.0, 1.0]),
}


def embed_molecule(molecule, tokenizer, encoder, args):
    output = tokenizer(
        molecule,
        max_length=args.tokenizer_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
    )
    embed = encoder(
        input_ids=output.input_ids.to(args.accelerator),
        attention_mask=output.attention_mask.to(args.accelerator),
    ).last_hidden_state
    return mean(embed, dim=1).cpu()


class GinsengT5RegressionInference(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.first_linear = nn.Linear(
            FEATURE_SHAPE[args.pretrained_name], args.hidden_dim
        )
        self.first_norm = nn.LayerNorm(args.hidden_dim)
        self.first_relu = nn.LeakyReLU()
        self.first_dropout = nn.Dropout(args.dropout)

        if args.use_adduct:
            self.adduct_linear = nn.Linear(2, args.hidden_dim)
            self.adduct_norm = nn.LayerNorm(args.hidden_dim)
            self.adduct_relu = nn.LeakyReLU()
            self.adduct_dropout = nn.Dropout(args.dropout)

        pre_dim = args.hidden_dim * 2 if args.use_adduct else args.hidden_dim
        for i, unit in enumerate(args.linears):
            setattr(self, f"linear_{i}", nn.Linear(pre_dim, unit))
            setattr(self, f"norm_{i}", nn.LayerNorm(unit))
            setattr(self, f"relu_{i}", nn.LeakyReLU())
            setattr(self, f"dropout_{i}", nn.Dropout(args.dropout))
            pre_dim = unit

        self.final_linear = nn.Linear(pre_dim, 1)

    def forward(self, embed, adduct):
        out1 = self.first_linear(embed)
        out1 = self.first_norm(out1)
        out1 = self.first_relu(out1)
        out1 = self.first_dropout(out1)

        if self.args.use_adduct:
            out2 = self.adduct_linear(adduct)
            out2 = self.adduct_norm(out2)
            out2 = self.adduct_relu(out2)
            out2 = self.adduct_dropout(out2)

        out = concat([out1, out2], dim=1) if self.args.use_adduct else out1

        for i in range(len(self.args.linears)):
            out = getattr(self, f"linear_{i}")(out)
            out = getattr(self, f"norm_{i}")(out)
            out = getattr(self, f"relu_{i}")(out)
            out = getattr(self, f"dropout_{i}")(out)

        out = self.final_linear(out)
        return out


class RegressionPredictor:
    def __init__(self, args_with_adduct, args_without_adduct):
        self.args_with_adduct = args_with_adduct
        self.args_without_adduct = args_without_adduct

        self.model_with_adduct = GinsengT5RegressionInference(args=args_with_adduct)
        self.model.load_state_dict(torch.load(args_with_adduct.checkpoint_path))
        self.model_with_adduct.eval()

        self.model_without_adduct = GinsengT5RegressionInference(
            args=args_without_adduct
        )
        self.model.load_state_dict(torch.load(args_without_adduct.checkpoint_path))
        self.model_without_adduct.eval()

    def get_features(self, molecule, pretrained_name=None):
        if pretrained_name == "mol2vec":
            feature = get_fingerprint(molecule, pretrained_name).to_numpy()
        elif pretrained_name == "estate":
            pretrained_name = PRETRAIN_MAPPING.get(pretrained_name, None)
            tmp_file = "tmp.txt"
            with open(tmp_file, "w") as f:
                f.write(molecule)
            ligand = iFeatureOmegaCLI.iLigand(tmp_file)
            ligand.get_descriptor(pretrained_name)
            feature = ligand.encodings
            feature = feature.iloc[0].to_numpy()
            os.remove(tmp_file)

        elif "molt5" in pretrained_name:
            # check if
            pretrained_name = PRETRAIN_MAPPING.get(pretrained_name, None)
            if pretrained_name is None:
                raise ValueError(f"Unsupported pretrained name: {pretrained_name}")

            if not hasattr(self, "tokenizer"):
                self.tokenizer = T5Tokenizer.from_pretrained(pretrained_name)
                self.encoder = T5EncoderModel.from_pretrained(pretrained_name).to(
                    self.args_with_adduct.accelerator
                )
            feature = (
                embed_molecule(
                    molecule, self.tokenizer, self.encoder, self.args_with_adduct
                )
                .squeeze(0)
                .detach()
                .numpy()
            )
        else:
            raise ValueError(
                f"Unsupported pretrained name: {self.args_with_adduct.pretrained_name}"
            )
        return feature

    def predict(self, molecule, adduct=None):
        if adduct is not None:
            pretrained_name = self.args_with_adduct.pretrained_name
        else:
            pretrained_name = self.args_without_adduct.pretrained_name
        feature = self.get_features(molecule, pretrained_name)
        feature_tensor = torch.tensor(feature).unsqueeze(0).float()
        if adduct is not None:
            adduct_tensor = torch.tensor(adduct).float()
            prediction = self.model_with_adduct(feature_tensor, adduct_tensor)
        else:
            prediction = self.model_without_adduct(feature_tensor, None)

        return prediction.item()

    def predict_batch(self, molecules, adducts=None, batch_size=2):
        if adducts is not None:
            pretrained_name = self.args_with_adduct.pretrained_name
        else:
            pretrained_name = self.args_without_adduct.pretrained_name
        print(f"Using pretrained name: {pretrained_name}")
        predictions = []
        for i in range(0, len(molecules), batch_size):
            batch_molecules = molecules[i : i + batch_size]
            batch_features = [
                self.get_features(mol, pretrained_name) for mol in batch_molecules
            ]
            print(f"Batch features shape: {[f.shape for f in batch_features]}")
            feature_tensor = torch.tensor(batch_features).float()

            if adducts is not None:
                batch_adducts = [
                    ADDUCT.get(adduct, torch.zeros(2))
                    for adduct in adducts[i : i + batch_size]
                ]
                adduct_tensor = torch.stack(batch_adducts).float()
                print(f"Adduct tensor shape: {adduct_tensor.shape}")
                preds = self.model_with_adduct(feature_tensor, adduct_tensor)
            else:
                preds = self.model_without_adduct(feature_tensor, None)

            predictions.extend(preds.squeeze(1).detach().cpu().numpy().tolist())

        return predictions


if __name__ == "__main__":
    import json
    import pandas as pd
    import argparse

    # ============================== Example for CCS ==============================
    # ccs_folder = "/home/ndhieunguyen/ginseng/GinDB-AI/regression/ckpt/CCS"
    # with_adduct_folder = os.path.join(ccs_folder, "with_adduct")
    # with_adduct_args = json.load(
    #     open(os.path.join(with_adduct_folder, "training_config.json"), "r")
    # )
    # with_adduct_args["checkpoint_path"] = os.path.join(with_adduct_folder, "model.ckpt")
    # # convert dict to argparse.Namespace
    # with_adduct_args = argparse.Namespace(**with_adduct_args)

    # without_adduct_folder = os.path.join(ccs_folder, "without_adduct")
    # without_adduct_args = json.load(
    #     open(os.path.join(without_adduct_folder, "training_config.json"), "r")
    # )
    # without_adduct_args["checkpoint_path"] = os.path.join(
    #     without_adduct_folder, "model.ckpt"
    # )
    # # convert dict to argparse.Namespace
    # without_adduct_args = argparse.Namespace(**without_adduct_args)

    # predictor = RegressionPredictor(with_adduct_args, without_adduct_args)

    # df = pd.read_csv("/home/ndhieunguyen/ginseng/GinDB-AI/regression/data/final_test/mol2vec.csv")
    # molecules = df["Canonical_Smiles"].tolist()
    # adducts = df["Adducts"].tolist()

    # predictions = predictor.predict_batch(molecules, adducts, batch_size=2)
    # print(f"Predictions: {predictions}")

    # ============================== Example for Retention Time ==============================
    tR_folder = "/home/ndhieunguyen/ginseng/GinDB-AI/regression/ckpt/tR"
    with_adduct_folder = os.path.join(tR_folder, "with_adduct")
    with_adduct_args = json.load(
        open(os.path.join(with_adduct_folder, "training_config.json"), "r")
    )
    with_adduct_args["checkpoint_path"] = os.path.join(with_adduct_folder, "model.ckpt")
    # convert dict to argparse.Namespace
    with_adduct_args = argparse.Namespace(**with_adduct_args)
    without_adduct_folder = os.path.join(tR_folder, "without_adduct")
    without_adduct_args = json.load(
        open(os.path.join(without_adduct_folder, "training_config.json"), "r")
    )
    without_adduct_args["checkpoint_path"] = os.path.join(
        without_adduct_folder, "model.ckpt"
    )
    # convert dict to argparse.Namespace
    without_adduct_args = argparse.Namespace(**without_adduct_args)
    predictor = RegressionPredictor(with_adduct_args, without_adduct_args)
    df = pd.read_csv(
        "/home/ndhieunguyen/ginseng/GinDB-AI/regression/data/final_test/estate.csv"
    )
    molecules = df["Canonical_Smiles"].tolist()
    adducts = None
    predictions = predictor.predict_batch(molecules, adducts, batch_size=2)
    print(f"Predictions: {predictions}")
