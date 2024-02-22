from glob import glob
from utils import read_file,get_labels
from tqdm import tqdm
from pathlib import Path

def cleanData(DATA_DIR):
    """
    Identifying the names of files which do not have labels and deleting those files from the directory

    Arguments:
      DATA_DIR: parent directory path for dataset

    Returns:
      Number of files deleted
    """

    IMAGE_DIR = DATA_DIR / "images"
    LABEL_DIR = DATA_DIR / "labels"

    #Storing paths to all images
    images = list(IMAGE_DIR.glob("*"))

    #Storing paths to all labels
    labels = []

    #Storing path to all labels
    for image_path in images:
      filename = image_path.name
      filename = ".".join(filename.split('.')[:-1])
      label_path = LABEL_DIR / f"{filename}.txt"
      labels.append(label_path)

    #Identifying files with no labels and eleting them
    delete_counter=0
    for idx in tqdm(range(len(labels))):
      temp_text = read_file(labels[idx])
      x,y = get_labels(temp_text)
      if x==[""]:
        labels[idx].unlink()
        images[idx].unlink()
        delete_counter = delete_counter + 1

    return delete_counter

BASE_DIR = Path.cwd()
TRAIN_DATA_DIR = BASE_DIR/ "resources" / "dataset" / "train"
VALID_DATA_DIR = BASE_DIR/ "resources" / "dataset" / "valid"
TEST_DATA_DIR = BASE_DIR/ "resources" / "dataset" / "test"


print("Number of files Deleted:",cleanData(TRAIN_DATA_DIR))
print("Number of files Deleted:",cleanData(VALID_DATA_DIR))
print("Number of files Deleted:",cleanData(TEST_DATA_DIR))