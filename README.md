# BRAIN TUMOR DETECTION

This project aims to build an object detection model to detect tumor from MRI images on this kaggle dataset. 
There are three labels to this dataset.

# Installation
Clone repo and install requirements.txt in a Python>=3.12.0

> git clone https://github.com/achyutk/brain_tumor_detection.git
> cd brain_tumor_detection.git <br>
> pip install -r requirements.txt    #install <br>

# Dataset
This code already provides with the resources/dataset needed to execute the code. 

However, if you wish to download the images for source file you can get it here : 
https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection/data


# Model Weights
The training.ipynb file when executed will generate model weights in the resources/output directory.  

If you want the model weights that I got from training, reach out to me. 

# Scripts:

> model.py:  code to build model for training. 

> dataset.py : code to build custom dataset and custom dataloader to use for training. 

> utils.py : code which contains helper function.