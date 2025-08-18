import os
import json
from torch.utils.data import DataLoader
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.data import GingsengDataset, prepare_dataset
from src.utils import get_time_string
from src.model import *
import wandb


def main(args):
    seed_everything(args.seed)
    if args.prepare_dataset:
        prepare_dataset(args)

    else:
        time_now = get_time_string()
        if args.k_folds > 0:
            result_dict = {}
            training_config = {}
            predictions = []
            for fold in range(args.k_folds):
                use_adduct = "adduct" if args.use_adduct else "no_adduct"
                use_pca = f"pca_{args.n_components}" if args.use_pca else "no_pca"
                data_folder = os.path.join(
                    "dataset",
                    "cross_val",
                    f'{args.target_col.replace(" ", "_").replace("/", "_")}',
                    f"{args.pretrained_name.replace('/', '_').replace('-', '_')}_{use_adduct}_{use_pca}",
                    str(fold),
                )
                train_dataset = GingsengDataset(
                    os.path.join(data_folder, "train.csv"), args
                )
                val_dataset = GingsengDataset(
                    os.path.join(data_folder, "val.csv"), args
                )
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=128,
                    shuffle=False,
                )
                model = eval(args.model_class)(args, fold, time_now)

                training_config = {
                    **args.__dict__,
                    "train_num_samples": len(train_dataset),
                    "val_num_samples": len(val_dataset),
                }
                if args.use_wandb:
                    wandb_logger = WandbLogger(
                        name=f"{args.pretrained_name}_{time_now}",
                        project=args.project,
                        log_model=True,
                        config=training_config,
                        settings=wandb.Settings(_disable_stats=True),
                    )
                    logger = wandb_logger
                else:
                    logger = False

                checkpoint_callback = ModelCheckpoint(
                    monitor="val/r2_score", mode="max", save_top_k=1
                )
                trainer = Trainer(
                    logger=logger,
                    max_epochs=args.epochs,
                    accelerator=args.accelerator,
                    devices=args.devices,
                    callbacks=[checkpoint_callback],
                    enable_progress_bar=True,
                    log_every_n_steps=1,
                    check_val_every_n_epoch=args.check_val_every_n_epoch,
                    enable_checkpointing=True,
                )
                trainer.fit(
                    model=model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                )

                for k, v in trainer.callback_metrics.items():
                    if k not in result_dict:
                        result_dict[k] = [v.item()]
                    else:
                        result_dict[k].append(v.item())

                gts_and_preds = trainer.predict(
                    model=model,
                    dataloaders=val_dataloader,
                )
                predictions.append(gts_and_preds[0])

                del model, trainer
                del train_dataloader, train_dataset
                del val_dataloader, val_dataset

            metrics_dict = {}
            for k, v in result_dict.items():
                metrics_dict[k] = sum(v) / len(v)

            use_adduct = "adduct" if args.use_adduct else "no_adduct"
            use_pca = "pca" if args.use_pca else "no_pca"
            result_folder = os.path.join(
                args.result_folder,
                f'{args.target_col.replace(" ", "_").replace("/", "_")}',
                (
                    args.pretrained_name.replace("/", "_").replace("-", "_")
                    + f"_{use_adduct}"
                    + f"_{use_pca}"
                ),
                f"{time_now}",
            )
            os.makedirs(result_folder, exist_ok=True)

            with open(
                os.path.join(result_folder, "metrics.json"),
                "w+",
            ) as f:
                json.dump(metrics_dict, f)
            with open(
                os.path.join(result_folder, "training_config.json"),
                "w+",
            ) as f:
                json.dump(training_config, f)

            with open(
                os.path.join(result_folder, "predictions.json"),
                "w+",
            ) as f:
                json.dump(predictions, f)

        else:
            training_config = {}
            use_adduct = "adduct" if args.use_adduct else "no_adduct"
            use_pca = f"pca_{args.n_components}" if args.use_pca else "no_pca"
            data_folder = os.path.join(
                "dataset",
                "full",
                f'{args.target_col.replace(" ", "_").replace("/", "_")}',
                f'{args.pretrained_name.replace("/", "_").replace("-", "_")}_{use_adduct}_{use_pca}',
            )
            train_dataset = GingsengDataset(
                os.path.join(data_folder, "train.csv"), args
            )
            test_dataset = GingsengDataset(os.path.join(data_folder, "test.csv"), args)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=128,
                shuffle=False,
            )
            model = eval(args.model_class)(args, None, time_now)
            training_config = {
                **args.__dict__,
                "train_num_samples": len(train_dataset),
                "test_num_samples": len(test_dataset),
            }
            if args.use_wandb:
                wandb_logger = WandbLogger(
                    name=f"{args.pretrained_name}_{time_now}",
                    project=args.project,
                    log_model=True,
                    config=training_config,
                    settings=wandb.Settings(_disable_stats=True),
                )
                wandb_logger.watch(model)
                logger = wandb_logger
            else:
                logger = False
            trainer = Trainer(
                logger=logger,
                max_epochs=args.epochs,
                accelerator=args.accelerator,
                devices=args.devices,
                enable_progress_bar=True,
                log_every_n_steps=1,
                check_val_every_n_epoch=args.check_val_every_n_epoch,
                enable_checkpointing=True,
            )
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
            )
            trainer.test(model=model, dataloaders=test_dataloader)

            use_adduct = "adduct" if args.use_adduct else "no_adduct"
            use_pca = "pca" if args.use_pca else "no_pca"
            result_folder = os.path.join(
                args.result_folder,
                f'{args.target_col.replace(" ", "_").replace("/", "_")}',
                (
                    args.pretrained_name.replace("/", "_").replace("-", "_")
                    + f"_{use_adduct}"
                    + f"_{use_pca}"
                ),
                f"{time_now}",
            )
            os.makedirs(result_folder, exist_ok=True)
            result_dict = {}
            for k, v in trainer.callback_metrics.items():
                result_dict[k] = v.item()

            gts_and_preds = trainer.predict(
                model=model,
                dataloaders=test_dataloader,
            )
            with open(
                os.path.join(result_folder, "metrics.json"),
                "w+",
            ) as f:
                json.dump(result_dict, f)
            with open(
                os.path.join(result_folder, "training_config.json"),
                "w+",
            ) as f:
                json.dump(training_config, f)
            with open(
                os.path.join(result_folder, "predictions.json"),
                "w+",
            ) as f:
                json.dump(gts_and_preds[0], f)

            trainer.save_checkpoint(os.path.join(result_folder, "model.ckpt"))


if __name__ == "__main__":
    import argparse
    from config import config

    parser = argparse.ArgumentParser()
    for k, v in config.__dict__.items():
        if type(v) in [str, int, float]:
            parser.add_argument(f"--{k}", type=type(v), default=v)
        elif type(v) == bool:
            parser.add_argument(f"--{k}", action="store_false" if v else "store_true")
        elif type(v) == list:
            parser.add_argument(f"--{k}", nargs="*", type=type(v[0]), default=v)

    args = parser.parse_args()
    main(args)
