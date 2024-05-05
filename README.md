# MAGNet

This is the Pytorch implementation of MAGNet in the paper: **Forecasting battery degradation trajectory under domain shift with domain generalization**.



# Requirements

Key dependencies:

- Ubuntu 20.04

- Python 3.8.13
- CUDA 11.1
- torch == 1.9.0
- matplotlib == 3.5.3
- jupyter == 1.0.0
- numpy == 1.23.4
- pandas == 1.4.4
- Wasserstein == 1.1.0

Dependencies can be installed using the following command:

```
pip install -r reqruiements.txt
```







# Reproducibility

## Easy reproducibility

To help the reproducibility, we have provided the trained model weights and other necessary files in the `Figshare`, thus you can easily see the main results in this paper. First of all, you need to download the `checkpoints`, `dataset`, `detailed_results`, `fig5` and `results`  folders from the `Google Drive` and then put them under the project root folder. After that,  you can reproduce the results using the following scripts:

- evaluate_model.py: You can evaluate the model performance using this Python script.

- Plot_fig3.ipynb: This notebook gives the draft fig. 3 of this paper.

- visualize_enc_new_conditionLevel.py: You can visualize the embeddings on the NCA dataset using this Python script.

- visualize_enc_new_conditionLevel_NE.py: You can visualize the embeddings on the LFP dataset using this Python script.

- Compare_PCA_and_mechanism_NCA.py: Compute the superiority matrix.

  

The datasets and pretrained model weights are released at Google Drive: https://drive.google.com/drive/folders/1iHu32xrkbNOzqbQM9qH-RXMdxxzj3TZm?usp=sharing

Note that the directory names for pretrained MAGNet weights are as follows:

- NCA dataset: `NCA_meta_Informer_Batteries_cycle_SLMove_lr1e-06_metalr0.005_mavg15_ftM_sl10_ll10_pl150_dm8_nh4_el1_dl2_df2_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_lossawmse_vallossnw_dp0.0_bs32_wd0_mb0.25_agamma0.25_lradjtype4_0`

- LFP dataset: `NE_meta_Informer_Batteries_cycle_SLMove_lr1e-06_metalr0.0075_mavg15_ftM_sl20_ll20_pl500_dm12_nh4_el2_dl2_df4_fc5_fc21_ebCycle_dtFalse_valratio0.5_test_lossawmse_vallossnw_dp0.0_bs128_wd0_mb2_agamma0.2_lradjtype4_0`

  

## Reproduce from the raw data

To reproduce the results, the same dependencies and core computing devices should be used. You can install the dependencies according to the `Requirements`. Additionally, you need a Ubuntu 20.04 machine with GTX 3090 GPUs. 

**Data processing**
Initially, you need to process the raw data to get the processed dataset. For the NCA dataset, you need to download the `NCA_data_process` from the `Google Drive`. After installing the dependencies shown in the `requirements.txt`, you can run the following scripts in order to get the processed dataset used in this work:

1. abnormal_utils.py
2. 2022NC_cycle_provider.py

For the LFP dataset, you need to download the `LFP_data_process` from the `Google Drive`.  After installing the dependencies shown in the `requirements.txt`,  you can run the following scripts in order to get the processed dataset used in this work:

1. BuildPkl_Batch1.ipynb; BuildPkl_Batch2.ipynb; BuildPkl_Batch3.ipynb
2. Load Data.ipynb
3. utils.py
4. process_data_Autoformer.ipynb

Then, you can put the processed datasets under the `dataset` folder.

**Train the model**

You can reproduce the results in this paper using the same hyperparameters listed in the supplementary information. The code for training models on the NCA and LFP datasets are `run_meta_NCA.py` and `run_meta_NE.py` respectively. The training might cost 20 hours to finish in the LFP dataset.

```bash
# Example
# The following instruction runs the code in background and saves the ouput in the log.out file.
nohup python -u run_meta_NE.py &> ./log.out&
nohup python -u run_meta_NCA.py &> ./log2.out&
```