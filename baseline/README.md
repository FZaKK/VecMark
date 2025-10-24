# VAE-based Approach

Our work adopt a VAE-based zero-watermarking scheme as baseline. The VAE-base model characterizes the vector space via vector reconstruction training.

## 🛠 Core Scripts

### 1. `train_vae.py`
- **Function**: Training script for VAE-based model.
- **Parameters**:
  - 🔄 `--train`: input file path
  - 🔄 `--epochs`: training epochs
  - 🔄 `--dim_emb`: input vector dimensions
  - 🔄 `--model_type`: model type (including: vae, dae, aae)
  - 🔄 `--lambda_kl`: KL divergence coefficient
  - 🔄 `--save-dir`: save path
  - 🔄 `--append-results`: appending results
- **Instruction**: 
  ```python
  python train_vae.py --train train_path --dim_emb 768 --model_type vae --lambda_kl 0.1 --save-dir save_path --epochs 50 --append-results
  ```



### 2. `test_vae.py`
- **Function**: Test the reconstruction loss distribution of two input vector sets.
- **Parameters**:
  - 🔄 File path for vector set 1
  - 🔄 File path for vector set 2
- **Instruction**:
  ```python
  python teat_vae.py
  ```


## 🔌 Notice

The file paths for both test vector sets can be manually specified.


<!--## ⚠️ Important Notice

Due to equipment failure, the following scripts are currently temporarily unavailable and will be added to the repository soon:

- sample.py (to be added shortly)
- judge.py (to be added shortly)
-->
