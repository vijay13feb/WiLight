## Title of the paper. 
# WiLight: Toward Lightweight and Embedded WiFi Sensing: Phase-Robust CSI Modeling for Human Activity Recognition

## 📥 Dataset
Download our dataset from the below link. 

🔗 **[WiLight Dataset](https://doi.org/10.5281/zenodo.11551205)**  

Download  Exposing the CSI dataset from the below link. 

🔗 **[Ax_Sense Dataset](https://github.com/ansresearch/exposing-the-csi)**  

Download  SimWiSense dataset from the below link. 

🔗 **[SimWiSense Dataset](https://ieee-dataport.org/documents/simwisense-wi-fi-csi-dataset-simultaneous-multi-subject-har)**

Download  SHARPac dataset from the below link. 

🔗 **[SHARPac Dataset](https://github.com/francescamen/SHARP)**


## 📂 Folder Structure
Maintain the folder structure like this 
```
WiLight/
├─ input_data/                                    
├─ Python_code/
│  ├─ Preprocessed/S1/
│  ├─ created_training_pruning.py
│  ├─ fine_tune_testing.py.py
│  └─ preprocessing_double_ratio_computation.py
|  └─ visualization.py
├─ TCN/
├─ requirements.txt
└─ README.md
```

## Prepare the Dataset
1. Download the dataset the above link
2. Preprocess the dataset and obtain the raw CSI data and delete the Pilot and Null subcarrier. 
3. Store the Raw CSI data files into ''/input_data'' using foler structure of the original dataset. 

## Preproces the Dataset
- After preparing the dataset create the ''/preprocessed'' folder inside the ''/Python_code'' folder and run the following script
```bash
python /Python_code/preprocessing_double_ratio_computation.py
```
- Note: Chnage the path of the input data files as per folder structure
---

## Initial Training and Pruning of the TCN model
```bash
python /Python_code/created_training_pruning.py
```

## Fine-tunning and Testing the TCN model

```bash
python /Python_code/fine_tune_testing.py
```

## Visualization of the Phase Double Ratio (PDR)
```bash
python /Python_code/visualization.py
```



