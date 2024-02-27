# CMS-Net
The repository contains an implementation of CMS-Net, which is a classification problem suitable for small sample datasets of the LC-MS type.
# Data preparation
There are obvious Batch effects in MDD data, which can be corrected using batch removal effect.R.
# Training and testing
The training and testing of the LSTM and CMS-Net models for the data set used in the study were carried out in mdd-net.py file, dl_model.py is the model file. The Transformer model is built in transformer.py
Machine learning uses the training and testing code of the five models in ML.py
# Suggested setup
More details can be found in requirements.txt.
