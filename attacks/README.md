# Several Attacks for Robustness Test

Our work mainly provides three forms of attacks to test the robustness of the vector database watermark. Among them, DSA is a brand - new form of attack.

## 🛠 Core Scripts

### 1. `dsa.py`
- **Function**: Randomly shift the dimensions in the vectors to be verified.
- **Parameters**:
  - 🔄 Input file path
  - 🔄 Save file path

### 2. `sea.py`
- **Function**: Randomly extract a subset of the vector set to be verified (from 1000 samples to 100).
- **Parameters**:
  - 🔄 Input file path
  - 🔄 Save file path

### 3. `gna.py`
- **Function**: Add random Gaussian noise to the vectors to be verified according to the distribution of the vector dimensions themselves.

- **Parameters**:
  - 🔄 Input file path
  - 🔄 Save file path

## 🔌 Notice

The file directories for input and output can be manually specified. All test codes can be executed using the following command.

```python
python dsa.py
python sea.py
python gna.py
```


<!--## ⚠️ Important Notice

Due to equipment failure, the following scripts are currently temporarily unavailable and will be added to the repository soon:

- sample.py (to be added shortly)
- judge.py (to be added shortly)
-->
