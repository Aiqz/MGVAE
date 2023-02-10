# MGVAE
> Code for "Generative Oversampling for Imbalanced Data via Majority-Guided VAE", AISTATS 2023.

## Usage
**For MGVAE training:**
```
python mgvae_train.py -c "your config file!"
```
Note that all config template of generative models used in our experiments are in "configs/configs_generation" directory, you need to add your data path for successfully run.
**For classifier trainig:**
```
python classifier_train.py -c "your config file!"
```
Note that all config template of classifiers used in our experiments are in "configs/configs_DATASET" directory.
