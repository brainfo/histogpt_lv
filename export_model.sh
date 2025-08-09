

module purge                              # start clean
module load cuda/11.8.0                   # OK – doesn’t touch sys.path
module load gcc/11.3.0                    # OK – same


# --- absolutely do NOT point PYTHONUSERBASE at the venv ---
unset PYTHONUSERBASE                      # or export PYTHONUSERBASE="" 
                                          # (and while we’re at it:)
unset PYTHONPATH                          # avoid stray paths from modules
export PYTHONNOUSERSITE=1                 # ignore ~/.local site-packages
source /proj/sens2022004/nobackup/wharf/hongki/hongki-sens2022004/histogpt_anne/histogpt_lt_env/bin/activate

cd /proj/sens2022004/nobackup/wharf/hongki/hongki-sens2022004/histogpt_anne/histogpt-lv-test

python export_inference_model.py \
--checkpoint ../histogpt-lv/results/best_model/fold_2_best_epoch=13_val_auroc=1.000.ckpt \
--histogpt_weights ../histogpt-l-6k-pruned.pt \
--output ../histogpt-bcc_cscc.pth