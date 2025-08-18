dropouts="0.0 0.05 0.1"
hidden_dims="16 32 64"
linears="16 32 64"
lrs="1e-2 1e-3 1e-4"
epochs="100 300 500"
batch_sizes="8 16 32"

for dropout in $dropouts; do
    for hidden_dim in $hidden_dims; do
        for linear in $linears; do
            for lr in $lrs; do
                for epoch in $epochs; do
                    for batch_size in $batch_sizes; do
                        CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "avalon" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "chemberta" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "ecfp4" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "ecfp6" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "estate" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "extended" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "fcfp4" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "fcfp6" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "fp2" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "hybridization" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "mol2vec" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "molt5_base" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "molt5_large" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "molt5_small" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "pubchem" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "rdkit" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "scibert" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "selformer" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
				CUDA_VISIBLE_DEVICES=0 python3 train.py \
                        --linears $linear \
                        --dropout $dropout \
                        --hidden_dim $hidden_dim \
                        --k_folds 10 \
                        --target_col "Observed CCS" \
                        --project "Ginseng CCS_no_adduct" \
                        --pretrained_name "standard" \
                        --lr $lr \
                        --epochs $epoch \
                        --batch_size $batch_size \
                        --num_workers 1  &
                        wait
                    done
                done
            done
        done
    done
done
