from torch.utils.data import Dataset
from torch import mean
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import os
import torch


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


def prepare_dataset(args):
    train_df = pd.read_csv(
        os.path.join(args.train_df_folder, f"{args.pretrained_name}.csv")
    )
    test_df = pd.read_csv(
        os.path.join(args.test_df_folder, f"{args.pretrained_name}.csv")
    )
    cols_to_remove = [
        "Canonical_Smiles",
        "Observed m_z",
        "Observed Retention time",
        "Observed CCS",
        "Adducts",
    ]
    cols = [col for col in train_df.columns if col not in cols_to_remove]
    train_df = train_df[cols + [args.target_col] + ["Adducts"]]
    test_df = test_df[cols + [args.target_col] + ["Adducts"]]
    use_adduct = "adduct" if args.use_adduct else "no_adduct"
    use_pca = f"pca_{args.n_components}" if args.use_pca else "no_pca"
    if args.k_folds > 0:
        kfold = KFold(args.k_folds, shuffle=True, random_state=args.seed)
        for fold, (_, valid_idx) in enumerate(
            kfold.split(train_df, train_df[args.target_col])
        ):
            train_df.loc[valid_idx, "fold"] = fold

        for fold in range(args.k_folds):
            test_df_copy = test_df.copy()
            train_df_split = train_df[train_df["fold"] != fold].reset_index(drop=True)
            val_df_split = train_df[train_df["fold"] == fold].reset_index(drop=True)

            if args.use_pca:
                pca = PCA(n_components=args.n_components)
                all_train_embed = torch.tensor(train_df_split[cols].to_numpy())
                all_val_embed = torch.tensor(val_df_split[cols].to_numpy())
                all_test_embed = torch.tensor(test_df_copy[cols].to_numpy())

                all_train_embed = [
                    embed for embed in pca.fit_transform(all_train_embed)
                ]
                all_val_embed = [embed for embed in pca.transform(all_val_embed)]
                all_test_embed = [embed for embed in pca.transform(all_test_embed)]

                for j in range(len(all_train_embed[0])):
                    column = []
                    for i in range(len(all_train_embed)):
                        column.append(all_train_embed[i][j])
                    train_df_split[f"pca_{j}"] = column

                for j in range(len(all_val_embed[0])):
                    column = []
                    for i in range(len(all_val_embed)):
                        column.append(all_val_embed[i][j])
                    val_df_split[f"pca_{j}"] = column

                for j in range(len(all_test_embed[0])):
                    column = []
                    for i in range(len(all_test_embed)):
                        column.append(all_test_embed[i][j])
                    test_df_copy[f"pca_{j}"] = column

                for col in cols:
                    if col in train_df_split.columns:
                        train_df_split.drop([col], axis=1, inplace=True)
                    if col in val_df_split.columns:
                        val_df_split.drop(columns=[col], inplace=True)
                    if col in test_df_copy.columns:
                        test_df_copy.drop(columns=[col], inplace=True)

            folder_path = os.path.join(
                "dataset",
                "cross_val",
                f'{args.target_col.replace(" ", "_").replace("/", "_")}',
                f'{args.pretrained_name.replace("/", "_").replace("-", "_")}_{use_adduct}_{use_pca}',
                str(fold),
            )
            os.makedirs(folder_path, exist_ok=True)
            train_df_split.to_csv(os.path.join(folder_path, "train.csv"), index=False)
            val_df_split.to_csv(os.path.join(folder_path, "val.csv"), index=False)
            test_df_copy.to_csv(os.path.join(folder_path, "test.csv"), index=False)
    else:
        if args.use_pca:
            pca = PCA(n_components=args.n_components)
            all_train_embed = torch.tensor(train_df[cols].to_numpy())
            all_test_embed = torch.tensor(test_df[cols].to_numpy())

            all_train_embed = [embed for embed in pca.fit_transform(all_train_embed)]
            all_test_embed = [embed for embed in pca.transform(all_test_embed)]

            for j in range(len(all_train_embed[0])):
                column = []
                for i in range(len(all_train_embed)):
                    column.append(all_train_embed[i][j])
                train_df[f"pca_{j}"] = column

            for j in range(len(all_test_embed[0])):
                column = []
                for i in range(len(all_test_embed)):
                    column.append(all_test_embed[i][j])
                test_df[f"pca_{j}"] = column

            for col in cols:
                if col in train_df.columns:
                    train_df.drop([col], axis=1, inplace=True)
                if col in test_df.columns:
                    test_df.drop(columns=[col], inplace=True)

        folder_path = os.path.join(
            "dataset",
            "full",
            f'{args.target_col.replace(" ", "_").replace("/", "_")}',
            f'{args.pretrained_name.replace("/", "_").replace("-", "_")}_{use_adduct}_{use_pca}',
        )
        os.makedirs(folder_path, exist_ok=True)
        train_df.to_csv(os.path.join(folder_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(folder_path, "test.csv"), index=False)


class GingsengDataset(Dataset):
    def __init__(self, df_path, args):
        super().__init__()
        self.args = args
        self.df = pd.read_csv(df_path)
        self.target_col = args.target_col
        self.adducts_dict = {
            "[M-H+HCOOH]-": torch.tensor([1.0, 0.0]),
            "[M-H]-": torch.tensor([0.0, 1.0]),
        }
        cols_to_remove = ["Adducts", "fold"] + [args.target_col]
        self.cols = [col for col in self.df.columns if col not in cols_to_remove]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.log(
            torch.tensor(self.df[self.target_col].iloc[idx]).to(torch.float32)
        )
        adducts = self.adducts_dict[self.df["Adducts"].iloc[idx]]
        features = torch.tensor(self.df[self.cols].iloc[idx].to_numpy()).to(
            torch.float32
        )

        return features, adducts, label
