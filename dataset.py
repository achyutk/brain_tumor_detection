from torch.utils.data import Dataset,DataLoader
from glob import glob
import os
import PIL
import torch
from torchvision import tv_tensors
from torchvision.ops.boxes import box_area
from torchvision import transforms
from tqdm.notebook import tqdm
from utils import read_file,get_labels,process_label
import references.detection.utils as ut

class CustomDataset(Dataset):
  def __init__(self,DATA_DIR,transforms = None):
    """
    Initialising variables -> parent directory path, list of image paths, list of lables path and transformation that can be done on images

    Arguments:
      rootDir: parent directory path for dataset
      transform: transformation that should be applied on the dataset

    Returns:
      None
    """

    self.DATA_DIR = DATA_DIR
    self.transforms = transforms

    IMAGE_DIR = DATA_DIR / "images"
    LABEL_DIR = DATA_DIR / "labels"
    
    #Storing paths to all images
    self.images = list(IMAGE_DIR.glob("*"))

    #Storing paths to all labels
    self.labels = []

    #Storing path to all labels
    for image_path in self.images:
      filename = image_path.name
      filename = ".".join(filename.split('.')[:-1])
      label_path = LABEL_DIR / f"{filename}.txt"
      self.labels.append(label_path)

  def __len__(self):
    """
    Calculates the length of the dataset

    Arguments:
      None
    Returns:
      int : length of the dataset
    """
    #Returning the length of the dataset
    return len(self.images)

  def __getitem__(self,index):
    """
    This function retuns an element from the dataset and information regarding it

    Arguments:
      index: Index of the image for which data needs to be extracted
    Returns:
      image: image in PIL image
      target: dictionary containg information of the labels of images. info contained: boxes position, labels, area of boxes, iscrowd and inex of image.
    """

    #Extracting image
    image_path = self.images[index]
    image = PIL.Image.open(image_path)
    h,w = image.size

    if self.transforms is not None:
      image = self.transforms(image)

    #Extracting lables
    label_path = self.labels[index]
    label_text = read_file(label_path)
    try:
      label,boxes = get_labels(label_text)
    except:
      print("Error at: ",label_path)
    label,boxes = process_label(label,boxes,h,w)

    #Calculating area of boxes
    area = box_area(boxes)
    #Setting image id
    image_id = index
    #Setting iscrowd to false for all objects
    iscrowd = torch.zeros((label.size(0), ), dtype=torch.int64)

    #Creating target dictionary
    target = {}
    target["boxes"] =  tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=[h,w])
    target["labels"] = label
    target["area"] =  area
    target["iscrowd"] = iscrowd
    target["image_id"] =  image_id

    transform = transforms.ToTensor()
    image = transform(image)

    return image,target

class CustomDataLoader:
  def __init__(self,dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn = ut.collate_fn):
    """
    Initialising variables -> data_loader

    Arguments:
      dataset: Object of Custom Dataset, 
      batch_size=1 : To set Batch size for data loader
      shuffle=True : To shuffle the dataset or not
      num_workers=4, 
      collate_fn = ut.collate_fn: How to collate the dataset 
    Returns:
      None
    """
    self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn )
    self.dataset = self.data_loader.dataset
  def __iter__(self):
    return iter(self.data_loader)  
  def __len__(self):
    """
    Function to return ength of a dataloader for an instance

    Arguments:
      self
    Returns:
      length of the dtaloader
    """
    return len(self.data_loader)
