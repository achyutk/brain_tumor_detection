# Add the parent directory to the system path
import model
from pathlib import Path
import torch
import PIL
from torchvision import transforms
from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes
import random

class Inference:
    def __init__(self):
        '''
        Instantiate your model class
        Arguments: None (Can change to the name of model later when multiple models different types are added models.py)
        Returns: None
        '''
        self.model = model.FastRCNN(num_classes=3).model
         
    def load_model(self, path):
        '''
        Function to load the model
        Arguments:
         path: path to model file
        Returns: None
        '''    
        # Load the model's state dictionary
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)

    def preprocess_image(self, image_path):
        '''
        Path to image
        Arguments: 
         image_path : path to image file
        Returns: image in tensor
        '''        
        image = PIL.Image.open(image_path)
        transform = transforms.ToTensor()
        input_image = transform(image)
        return [input_image]

    def predict(self, input_image):
        '''
        Create Predictions on the dataset
        Arguments: 
         inupt_image: image in tensor format
        Returns: 
         output: predictions from the model
        '''        
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_image)
        return output
    
    def get_bigger_box(self,box1,box2):
        x_0,y_0,x_1, y_1 = min(box1[0],box2[0]),min(box1[1],box2[1]),max(box1[2],box2[2]), max(box1[3],box2[2])
        return [x_0,y_0,x_1, y_1]

    def get_highest_iou_box(self,predicted_boxes):
         
        # Calculate IoU matrix
        iou_matrix = box_iou(predicted_boxes, predicted_boxes)
        iou_matrix.fill_diagonal_(0)

        # Find the indices of the maximum IoU values along rows and columns
        max_iou_indices = torch.argmax(iou_matrix)
        # Convert flat index to row and column indices
        row_idx, col_idx = divmod(max_iou_indices.item(), iou_matrix.size(1))
        box1 = predicted_boxes[row_idx].tolist()
        box2 = predicted_boxes[col_idx].tolist()
        return [box1,box2]

    def boxes_with_highest_iou(self,predictions):
        highest_iou_box = []
        highest_iou_label = []
        predicted_labels = predictions[0]['labels']
        predicted_boxes = predictions[0]['boxes']

        for label in torch.unique(predicted_labels):
            mask = (predicted_labels == label).nonzero().view(-1)
            if len(mask)>2:
                #Find boxes with highesh iou and caluclate the highest IOU Box
                boxes = self.get_highest_iou_box(predicted_boxes[mask])
                high_box = self.get_bigger_box(boxes[0],boxes[1]) 
                highest_iou_box.append(high_box)
                highest_iou_label.append(label.item())               
            elif len(mask)==2:
                #Calculate box with highest iou, No need to find the box with highest iou
                boxes = predicted_boxes[mask].tolist()
                high_box = self.get_bigger_box(boxes[0],boxes[1]) 
                highest_iou_box.append(high_box)
                highest_iou_label.append(label.item())
            else:
                #highest IOU box is itself
                high_box = predicted_boxes[mask].tolist()
                highest_iou_box.append(high_box[0])
                highest_iou_label.append(label.item())
        return torch.tensor(highest_iou_box),list(map(str,highest_iou_label))
    
    def get_image(self,image,boxes,labels):
        if len(labels)!=0:
            bounded_img = draw_bounding_boxes(image,boxes=boxes,labels=labels,colors=["red","green","yellow"])
            #Converting tensor to image
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(bounded_img)
        else:
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(image)
        return pil_image


#Initializing the path
BASE_DIR = Path.cwd()/"resources"
MODEL_DIR = BASE_DIR/"output"
TEST_DATA_DIR = BASE_DIR/"dataset"/"test"/"images"
VIZ_DIR = Path.cwd()/"inference"/"visualizations"

#Initializing the name of model
model_name="faster_rcnn_1_24_02_22_14_20_46.pth"
model_path = MODEL_DIR/model_name

#Getting path of images to make Predictions
image_paths = list(TEST_DATA_DIR.glob("*"))

#Iterating over 10 images to make Predictions saving the imgaes
for image_path in random.sample(image_paths, 10):

    # Intialialising instance of inference class
    inference_model = Inference()

    # Load the model
    loaded_model = inference_model.load_model(model_path)

    # Preprocess an image for prediction
    input_image = inference_model.preprocess_image(image_path)

    # Make predictions
    predictions = inference_model.predict(input_image)
    #Get boxes and labels with highes iou
    boxes,labels = inference_model.boxes_with_highest_iou(predictions)
    pred_input_image = (input_image[0]*255).to(torch.uint8)
    predicted_image = inference_model.get_image(pred_input_image,boxes,labels)
    predicted_image.save(VIZ_DIR/f"prediction_{image_path.name}")

