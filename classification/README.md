# GinDB-AI (Biological activity classification)

# Install environments
```zsh
conda env create -f env.yml
```

# Download features (for training) & checkpoints (for inferencing)
- [Features](https://1drv.ms/u/c/fa72f5f3c0e55162/EVyvO3806Q1Bt8FdI5i7wFUB5DZ3Vxrc0mhfxWhXtRGLTQ?e=vMhg0T)
- [Checkpoints](https://1drv.ms/u/c/fa72f5f3c0e55162/EZK5BwFqk2tMhflbxEK8DkYBpDrXtOjRtgJnJ4fZdplHTQ?e=dmekYO)

# Train models
```zsh
usage: train.py [-h] --alpha_focal ALPHA_FOCAL [ALPHA_FOCAL ...] --beta_focal BETA_FOCAL --number_of_layers NUMBER_OF_LAYERS --norm_type NORM_TYPE [--train_no_func_yes_func] --feature_name FEATURE_NAME [--pca_dimen PCA_DIMEN] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--k_fold K_FOLD] [--embed_dim EMBED_DIM] [--dropout DROPOUT] [--num_classes NUM_CLASSES]

options:
  -h, --help            show this help message and exit
  --alpha_focal ALPHA_FOCAL [ALPHA_FOCAL ...]
  --beta_focal BETA_FOCAL
  --number_of_layers NUMBER_OF_LAYERS
  --norm_type NORM_TYPE
  --train_no_func_yes_func
  --feature_name FEATURE_NAME
  --pca_dimen PCA_DIMEN
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --k_fold K_FOLD
  --embed_dim EMBED_DIM
  --dropout DROPOUT
  --num_classes NUM_CLASSES
```

# Inference models
```python
checkpoint_dir = "<ckpt_path>"
inference = create_inference_instance(checkpoint_dir)

# Test sample
print(inference.predict_sequences_batch(["C=C(C)C(O)CCC(C)(OC1OC(COC2OCC(O)C(O)C2O)C(O)C(O)C1O)C1CCC2(C)C1C(O)CC1C3(C)CC(O)C(OC4OC(CO)C(O)C(O)C4OC4OC(CO)C(O)C(O)C4O)C(C)(C)C3CCC12C", "CCCCCCC[C@H](O[H])/C([H])=C([H])/C#CC#C[C@@H](O[H])C=C"]))
```