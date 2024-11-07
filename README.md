# PETformer 

Welcome to the repository of PETformer: "PETformer: Long-term Time Series Forecasting via Placeholder-enhanced Transformer." We will officially release this repository following the acceptance of our paper.

## Key Features

1. Shared Placeholder: PETformer uses a shared placeholder that occupies the output window to be predicted. This placeholder is entered into the Transformer encoder for feature learning.

2. Long Sub-sequence Division: To enhance the semantic richness of Transformer tokens, PETformer employs a Long Sub-sequence Division strategy.

3. Information-Interaction Modes: PETformer explores several information-interaction modes based on a channel-independent strategy.

4. Token-wise Prediction Layer: A token-wise prediction layer is utilized in PETformer, significantly reducing the number of learnable parameters.

## Getting Started

### Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
conda create -n PETformer python=3.8
conda activate PETformer
pip install -r requirements.txt
```

### Data Preparation

All the datasets needed for PETformer can be obtained from the [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. Create a separate folder named ```./dataset``` and place all the CSV files in this directory.

### Training Example

You can easily reproduce the results from the paper by running the provided script command. For instance, to reproduce the main results, execute the following command:

```
sh run_main.sh
```

Similarly, you can reproduce the results of the ablation learning by using other instructions:

```
sh run_ablation_placeholder.sh
```

Additionally, you can specify separate scripts to run independent tasks, such as obtaining results on ethh1:

```
sh script/PETformer/etth1.sh
```

## Acknowledgement

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases and datasets:

https://github.com/yuqinie98/patchtst

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai
