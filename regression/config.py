import os


class BaseConfig:
    def __init__(self):
        # Model Config
        self.cache_dir = os.path.abspath(os.path.join("cache"))
        self.pretrained_name = "laituan245/molt5-base"
        self.model_class = "GinsengT5Regression"
        self.tokenizer_max_length = 256
        self.hidden_dim = 16
        self.linears = [16]
        self.dropout = 0.0
        self.use_adduct = False

        # Data Config
        self.train_df_folder = os.path.join("dataset", "final_train")
        self.test_df_folder = os.path.join("dataset", "final_test")
        self.target_col = ""

        # Training Config
        self.project = "Ginseng Regression Final"
        self.seed = 42
        self.lr = 1e-2
        self.lr_step_size = 50
        self.lr_gamma = 0.9
        self.epochs = 500
        self.check_val_every_n_epoch = 1
        self.batch_size = 128
        self.patience = 10
        self.checkpoint_path = ""
        self.accelerator = "cuda"
        self.devices = 1
        self.prepare_dataset = False
        self.root_dir = "checkpoints"
        self.k_folds = 10
        self.use_wandb = False
        self.num_workers = 4
        self.result_folder = os.path.abspath(os.path.join("results"))


config = BaseConfig()
