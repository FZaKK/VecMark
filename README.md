# VecMark
This repository is for our new work: "VecMark: A Robust Zero-Watermarking Scheme for Vector Databases via Diffusion Models", feel free to propose your issues!! 😎

## Abstract
Vector Databases (VDBs) have played a pivotal role in modern AI systems by enabling efficient management of high-dimensional vectors from unstructured data. The wide adoption of VDBs also raises serious concerns about copyright infringement, particularly in data publishing scenarios. Zero-watermarking stands out as an ideal method that allows ownership authenticity without modifying the original data. However, the absence of relational structure and explicit features (compared with relational databases and multimedia data) makes it difficult to migrate existing zero-watermarking techniques for VDBs. Thus, we introduce VecMark, a zero-watermarking scheme driven by the distribution features of vector space extracted from vector sets via customized diffusion models. To the best of our knowledge, VecMark is the first scheme designed for VDBs. Considering the unique feature of vectors stored in VDBs (the independence among elements within the vectors), we further propose the SWORD (**S**ecure **W**orkflow for **O**bfuscation-**R**esilient **D**ata Preprocessing) mechanism to enhance the robustness of zero-watermark verification. SWORD interacts with VecMark to enable verification even when the dimensions of the vectors are randomly obfuscated. Moreover, we also investigate the practicality of VecMark in multi-user scenarios. Extensive experimental results demonstrate that VecMark enables zero-watermark verification across different dimensions, with p-values in hypothesis testing differing by up to $E^{100}$, and is resilient against various forms of attacks. With training on merely 5,000 vectors, the diffusion model can sufficiently capture the features of the vector space and perform verification without original VDBs. Meanwhile, the diffusion model obtained from a single training session exhibits generalization capability, making it applicable for verification across different vector sets. 


## Getting Started

### Environment

`requirements.txt` is available:
```bash
pip install -r requirements.txt 
```


### Preparing Datasets

For diffusion model's training, we use the WikiText dataset from Huggingface. 

```python
from datasets import load_dataset

train_dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
```

For verification, we use the released datasets of repo [BEIR](https://github.com/beir-cellar/beir), you can get to know the detailed information in the corresponding repo. 

| Dataset ID | Corpus | Source Link |
|----|----|----|
| **`NFCorpus`** | 3.6K | [View Dataset](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)       |
| **`SCIDOCS`** | 25K |  [View Dataset](https://allenai.org/data/scidocs) |
| **`Quora`**  | 523K |  [View Dataset](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) |


## VecMark




## SWORD

## Running SWORD

To run the SWORD module, please configure the following parameters before execution:

1. Set Clustering and Bin Parameters  
   Define the parameters (modifiable as needed):  
   ```python
   cluster = 20
   bin = 50
2. Specify the Training Data Path   
   Set the path to the training data:
   ```python
   train_path = "path/to/train/data"
   ```
3. Specify the Testing Data Path   
   Set the path to the testing data under DSA that will be used for SWORD evaluation:
   ```python
   test_path = "path/to/test/data"
   ```
4. Set the Output Path   
   The recovered (dimension-restoration) data will be saved in `.pkl` format. Define the save path as follows:
   ```python
   save_path = "path/to/save/result.pkl"
   ```
5. Run the Script   
   After updating all paths and parameters in the python file, simply execute:
   ```bash
   python SWORD.py
   ```


## ❤️Acknowledgments

Our code of VAE-based approach is based on the work of [VAE](https://github.com/shentianxiao/text-autoencoders).
