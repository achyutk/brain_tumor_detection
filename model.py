from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from references.detection.engine import train_one_epoch, evaluate
from pathlib import Path
from datetime import datetime

class FastRCNN():
    def __init__(self, num_classes,train_data_loader,val_data_loader):
        """
        Initialising model -> Set model to FasterRCNN

        Arguments:
        num_classes: Defines the number of labels
        train_data_loader: 

        Returns:
        Non
        """

        # train on the GPU or on the CPU, if a GPU is not available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        #Defining the model
        self.model = self.get_model(num_classes)
        self.model.to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]

        self.initialize_optimizer(params)

        # Data loaders
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.OUT_DIR = Path.cwd()/"resources"/"output"

    def get_model(self,num_classes):
        
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        return model
    
    def initialize_optimizer(self,params):

        # Construct an optimizer
        self.optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.1
        )

    def train(self, num_epochs=2, print_freq=10):
        for epoch in range(num_epochs):
            # Train for one epoch, printing every `print_freq` iterations
            train_one_epoch(self.model,self.optimizer,self.train_data_loader,self.device,epoch ,print_freq)
            # Update the learning rate
            self.lr_scheduler.step()
        
            # Evaluate on the train dataset
            print(f"Train results for epoch {epoch}:")
            self.evaluate_result(self.train_data_loader)

            # Evaluate on the train dataset
            print(f"Validation results for epoch {epoch}:")
            self.evaluate_result(self.val_data_loader)

            #Saving the model
            path = f"faster_rcnn_{epoch}_"
            self.save_model(path)


        print("Training completed.")

    def evaluate_result(self,data_loader):
        return evaluate(self.model,data_loader,device=self.device)
    
    def save_model(self,path):
        ## Get the current date and time
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%y_%m_%d_%H_%M_%S") 
        model_save_path = path + formatted_datetime +".pth"
        #Saving the model
        torch.save(self.model.state_dict(),self.OUT_DIR/model_save_path)