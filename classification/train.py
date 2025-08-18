import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from focal_loss import focal_loss as FL
import glob, os
import copy
from torch import nn
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math
from architectures import SimpleNeuralNetwork
from sklearn.metrics import roc_auc_score

pl.seed_everything(42)

class MyTestCallback(pl.Callback):
    def __init__(self):
        self.test_results = None

    def on_test_end(self, trainer, pl_module):
        self.test_results = pl_module.on_test_end()

def calculate_tn_fp_fn_tp(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp

def calculate_mcc(TN, FP, FN, TP):
    mcc_score = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return mcc_score

class CustomDataset(Dataset):
    def __init__(self, ids, data_dir, is_train_yes_func_no_func):
        self.ids = ids
        self.data_dir = data_dir
        self.data, self.labels = self.load_data_and_labels(is_train_yes_func_no_func)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        embeddings = self.data[index]
        label = self.labels[index]
        return embeddings, label

    def load_data_and_labels(self, is_train_yes_func_no_func):
        data = []
        labels = []
        for id in self.ids:
            file_path = f"{self.data_dir}/{id}.pt"
            data.append(torch.load(file_path)['Embeddings'])
            
            label = torch.load(file_path)['Label']
            
            if is_train_yes_func_no_func:
                if label == 1 or label == 2:
                    label = 1
            else:
                if label == 1:
                    label = 0
                elif label == 2:
                    label = 1
                    
            labels.append(label)
        return data, labels

class MyModel(pl.LightningModule):
    def __init__(self, args, num_classes, embeddings_dim):
        super().__init__()
        self.num_classes = num_classes
        self.args = args
        self.loss_fn = FL(alpha=self.args.alpha_focal, gamma=self.args.beta_focal, device='cuda')
        self.model = SimpleNeuralNetwork(args, num_classes, embeddings_dim)
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def on_validation_epoch_start(self):
        self.val_y_preds = []
        self.val_y_gts = []
        self.val_y_probs = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y_hat = self.forward(x)
        y_prob = y_hat.softmax(dim=-1)[:, 1]
        y_pred = torch.argmax(y_hat, dim=1)
        
        # append to overall list
        self.val_y_preds.extend(y_pred.cpu().numpy().tolist())
        self.val_y_gts.extend(y.cpu().numpy().tolist())
        self.val_y_probs.extend(y_prob.cpu().numpy().tolist())

    def on_validation_epoch_end(self):
        accuracy = accuracy_score(self.val_y_gts, self.val_y_preds)
        recall = recall_score(self.val_y_gts, self.val_y_preds)
        precision = precision_score(self.val_y_gts, self.val_y_preds)
        f1 = f1_score(self.val_y_gts, self.val_y_preds)
        tn, fp, fn, tp = calculate_tn_fp_fn_tp(self.val_y_gts, self.val_y_preds)
        specificity = tn / (tn + fp)
        mcc = calculate_mcc(tn, fp, fn, tp)
        auc = roc_auc_score(self.val_y_gts, self.val_y_probs)
        
        val_results = {
            'val_accuracy': accuracy,
            'val_recall': recall,
            'val_specificity': specificity,
            'val_precision': precision,
            'val_f1': f1,
            'val_tn': tn,
            'val_fp': fp,
            'val_fn': fn,
            'val_tp': tp,
            'val_mcc': mcc,
            'val_auc': auc
        }
        self.log_dict(val_results)
        
        return val_results
        
    def on_test_start(self):
        self.test_y_preds = []
        self.test_y_gts = []
        self.test_y_probs = []
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y_hat = self.forward(x)
        y_prob = y_hat.softmax(dim=-1)[:, 1]
        y_pred = torch.argmax(y_hat, dim=1)
        
        # append to overall list
        self.test_y_preds.extend(y_pred.cpu().numpy().tolist())
        self.test_y_gts.extend(y.cpu().numpy().tolist())
        self.test_y_probs.extend(y_prob.cpu().numpy().tolist())
        
    def on_test_end(self):
        accuracy = accuracy_score(self.test_y_gts, self.test_y_preds)
        f1 = f1_score(self.test_y_gts, self.test_y_preds)
        recall = recall_score(self.test_y_gts, self.test_y_preds)
        precision = precision_score(self.test_y_gts, self.test_y_preds)
        tn, fp, fn, tp = calculate_tn_fp_fn_tp(self.test_y_gts, self.test_y_preds)
        specificity = tn / (tn + fp)
        mcc = calculate_mcc(tn, fp, fn, tp)
        auc = roc_auc_score(self.test_y_gts, self.test_y_probs)
        
        test_results = {
            'test_accuracy': accuracy,
            'test_f1': f1,
            'test_recall': recall,
            'test_specificity': specificity,
            'test_precision': precision,
            'test_tn': tn,
            'test_fp': fp,
            'test_fn': fn,
            'test_tp': tp,
            'test_mcc': mcc,
            'test_auc': auc
        }
        # self.log_dict(test_results)
        
        return test_results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
def test_model_avg_prob(model, ckpt_dirpath, test_loader):
    MODELS_LIST = []
    for pt_file in os.listdir(ckpt_dirpath):
        if not pt_file.endswith('.ckpt'):
            continue
        pt_file_path = os.path.join(ckpt_dirpath, pt_file)
        state_dict = torch.load(pt_file_path)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        model.to('cuda')
        model = model.float()
        
        MODELS_LIST.append(copy.deepcopy(model))
    
    ALL_Y_HAT = []
    ALL_Y = []
    ALL_Y_PROB = []
    for batch in test_loader:
        total_prob_batch = []
        x, y = batch
        x = x.to('cuda')
        y = y.to('cuda')
        x = x.float()
        for model in MODELS_LIST:
            with torch.no_grad():
                output_prob = model(x)
                
            total_prob_batch.append(output_prob.softmax(dim=-1))
            
        avg_prob_batch = torch.sum(torch.stack(total_prob_batch), dim=0) / len(MODELS_LIST)
        
        y_hat = torch.argmax(avg_prob_batch, dim=-1)
        y_prob = avg_prob_batch[:, 1]
        ALL_Y_HAT.append(y_hat)
        ALL_Y.append(y)
        ALL_Y_PROB.append(y_prob)
        
    ALL_Y_HAT = torch.concat(ALL_Y_HAT)
    ALL_Y = torch.concat(ALL_Y)
    ALL_Y_PROB = torch.concat(ALL_Y_PROB)
    
    accuracy = accuracy_score(ALL_Y.cpu().numpy(), ALL_Y_HAT.cpu().numpy())
    f1 = f1_score(ALL_Y.cpu().numpy(), ALL_Y_HAT.cpu().numpy())
    recall = recall_score(ALL_Y.cpu().numpy(), ALL_Y_HAT.cpu().numpy())
    precision = precision_score(ALL_Y.cpu().numpy(), ALL_Y_HAT.cpu().numpy())
    tn, fp, fn, tp = calculate_tn_fp_fn_tp(ALL_Y.cpu().numpy(), ALL_Y_HAT.cpu().numpy())
    specificity = tn / (tn + fp)
    mcc = calculate_mcc(tn, fp, fn, tp)
    auc = roc_auc_score(ALL_Y.cpu().numpy(), ALL_Y_PROB.cpu().numpy())
    save_confusion_image(tn, fp, fn, tp, ckpt_dirpath, 'test_confusion_matrix.png')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'specificity': specificity,
        'precision': precision,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'mcc': mcc,
        'auc': auc
    }

def save_confusion_image(tn, fp, fn, tp, saved_path, image_name):
    # Create the confusion matrix
    confusion_matrix = [[tn, fp],
                        [fn, tp]]

    # Create the heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Save the image to a path
    plt.savefig(os.path.join(saved_path, image_name))
                

def train_model(args, data_dir, train_ids, test_ids):
    # Load data
    train_dataset = CustomDataset(train_ids, data_dir, args.train_no_func_yes_func)
    test_dataset = CustomDataset(test_ids, data_dir, args.train_no_func_yes_func)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Perform cross-validation
    kfold = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=42)
    ckpt_dirpath = f"checkpoints_{'NO_FUNC_YES_FUNC' if args.train_no_func_yes_func else 'SINGLE_MULTI'}/{args.feature_name}_{args.alpha_focal}_{args.beta_focal}_{args.number_of_layers}_{args.norm_type}_{args.dropout}"
    
    all_fold_valid_best_result = {}
    all_fold_train_best_result = {}
    test_callback = MyTestCallback()
    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_dataset.data, train_dataset.labels)):
        
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dirpath,
            filename="fold_" + f"{fold}" + "_{epoch:03d}_{val_mcc:.2f}",
            monitor="val_mcc",
            mode="max",
            save_top_k=1
        )

        # Create model and trainer
        model = MyModel(args, args.num_classes, args.embed_dim)
        trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", callbacks=[test_callback, checkpoint_callback, LearningRateMonitor()])

        # Train model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Evaluate on validation set
        trainer.test(model, dataloaders=val_loader, ckpt_path='best')
        eval_result = test_callback.test_results
        save_confusion_image(eval_result['test_tn'],
                             eval_result['test_fp'],
                             eval_result['test_fn'],
                             eval_result['test_tp'],
                             ckpt_dirpath,
                             f'eval_confusion_matrix_fold_{fold}.png')
        
        for eval_metric_name in eval_result:
            if eval_metric_name not in all_fold_valid_best_result:
                all_fold_valid_best_result[eval_metric_name] = [eval_result[eval_metric_name]]
            else:
                all_fold_valid_best_result[eval_metric_name].append(eval_result[eval_metric_name])
        
        # Evaluate on train set
        trainer.test(model, dataloaders=train_loader, ckpt_path='best')
        train_result = test_callback.test_results
        save_confusion_image(train_result['test_tn'],
                             train_result['test_fp'],
                             train_result['test_fn'],
                             train_result['test_tp'],
                             ckpt_dirpath,
                             f'train_confusion_matrix_fold_{fold}.png')

        for train_metric_name in train_result:
            if train_metric_name not in all_fold_train_best_result:
                all_fold_train_best_result[train_metric_name] = [train_result[train_metric_name]]
            else:
                all_fold_train_best_result[train_metric_name].append(train_result[train_metric_name])
    
    with open(os.path.join(ckpt_dirpath, 'result_valid.txt'), 'w') as f:
        for eval_metric_name in all_fold_valid_best_result:
            f.write(f"{eval_metric_name}: {sum(all_fold_valid_best_result[eval_metric_name]) / len(all_fold_valid_best_result[eval_metric_name])}\n") 
    
    test_results = test_model_avg_prob(model, ckpt_dirpath, test_loader)
    
    with open(os.path.join(ckpt_dirpath, 'result_test.txt'), 'w') as ff:
        for test_metric_name in test_results:
            ff.write(f"{test_metric_name}: {test_results[test_metric_name]}\n")
    
    
    with open(os.path.join(ckpt_dirpath, 'result_train.txt'), 'w') as f:
        for train_metric_name in all_fold_train_best_result:
            f.write(f"{train_metric_name}: {sum(all_fold_train_best_result[train_metric_name]) / len(all_fold_train_best_result[train_metric_name])}\n") 
    
        
        
def load_ids(txt_file):
    with open(txt_file, 'r') as f:
        ids_list = f.readlines()
    
    for i in range(len(ids_list)):
        ids_list[i] = ids_list[i].strip()
    
    return ids_list

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--alpha_focal', nargs="+", type=float, required=True)
    parser.add_argument('--beta_focal', type=int, required=True)
    parser.add_argument('--number_of_layers', type=int, required=True)
    parser.add_argument('--norm_type', type=str, required=True)
    parser.add_argument('--train_no_func_yes_func', action='store_true')
    parser.add_argument('--feature_name', type=str, required=True)
    parser.add_argument('--pca_dimen', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--k_fold', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=2)
    
    args = parser.parse_args()
    
    if args.train_no_func_yes_func:
        train_ids = load_ids('fixed_data_ids_new/fixed_train_ids.txt')
        test_ids = load_ids('fixed_data_ids_new/fixed_test_ids.txt')
    else:
        train_ids = load_ids('fixed_data_ids_new/fixed_train_ids_single_multi.txt')
        test_ids = load_ids('fixed_data_ids_new/fixed_test_ids_single_multi.txt')
    
    data_dir = os.path.join('features', args.feature_name)
    
    train_model(args, data_dir, train_ids, test_ids)