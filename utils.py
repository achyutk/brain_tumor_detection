import torch

def read_file(path):
    with open(path, "r") as f:
        return f.read()


def get_labels(text):
    #Extracting labels for labels file
    n_labels = text.split('\n')
    label=[]
    boxes=[]
    for row in n_labels:
        values = row.split(' ')
        label.append(values[0])
        boxes.append(list(map(float,values[1:])))
    return label,boxes

def process_label(label,boxes,h,w):
    processed_box = []
    for box in boxes:
        x_center, y_center, width, height = box
        x0 = w*(x_center - width/2)
        x1 = w*(x_center + width/2)
        y0 = h*(y_center - height/2)
        y1 = h*(y_center + height/2)
        processed_box.append([x0,y0,x1,y1])

    #Converting box info to tensor
    processed_box = torch.tensor(processed_box)
    processed_label = torch.tensor(list(map(int, label)))

    return processed_label, processed_box