import sys
sys.path.append('..')
import PIL
import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from utils import read_file, get_labels, process_label
from pathlib import Path
from glob import glob
import random


def visualise(image_path,label_path):
  """Function to Visualise the dataset
  Arguments:
      image_path: path for the image
      label_path: path for the label

    Returns:
      None
  """
  image = PIL.Image.open(image_path)
  h,w  = image.size

  labels = []
  label_text = read_file(label_path)
  labels,boxes = get_labels(label_text)
  labels,boxes = process_label(labels,boxes,h,w)
  labels = labels.tolist()
  labels = list(map(str,labels))
  # #Converting image to tensor
  transform = transforms.ToTensor()
  image = transform(image)
  image = (image*255).to(torch.uint8)

  # Creating an image
  bounded_img = draw_bounding_boxes(image,boxes=boxes,labels=labels,colors=["red","green","yellow"])

  #Converting tensor to image
  to_pil = transforms.ToPILImage()
  pil_image = to_pil(bounded_img)

  return pil_image
  
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR/ "resources" / "dataset" / "train"
IMAGE_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"
VIZ_DIR = BASE_DIR / "visualize" /"visualization"
# Use glob to get a list of images
images = list(IMAGE_DIR.glob("*"))

#Iterating over 100 images and visualising the
for image_path in  random.sample(images, 100):
    filename = image_path.name
    filename = ".".join(filename.split('.')[:-1])
    label_path = LABEL_DIR / f"{filename}.txt"
    pil_image = visualise(image_path,label_path)
    pil_image.save(VIZ_DIR / f"{filename}.jpg")