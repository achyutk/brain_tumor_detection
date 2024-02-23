import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path

class FastRCNN():
    def __init__(self, num_classes):
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