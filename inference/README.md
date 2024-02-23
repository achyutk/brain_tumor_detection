# Inference

This folder is used for infering the results from the model. 

Additionally, the code ignores boxes with high overlap and only considers boxes with highest IoU (i.e, If the model predicts multiple bounding boxes for a lable, it will identify a boxes with highest IoU and will create a new box overlapping both)



To execute the scripts in here, make sure the resources/output folder has the model weights file inoder to generate images.
> Either you can train the model using training.ipynb file
> Or request me for model wights.

# Scripts:

> inference.py: file to predict and generate images with bounding box for random 10 images from the testing dataset folder

> model.py : Used to create instance of the model in inference.py

# Results:

Images with predicted bounding boxes will be generated in visualizations folder
