#dataset.py
import torch, os
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def remove_irrelivent(directory):
    """
    This function just removed all files with the string "superpixels" in their
    name from a directory

    Returns
    -------
    None.

    """
    for f in os.scandir(directory):
        print("looking at " , f.name)
        if "superpixels" in f.name:
            os.remove(directory+"/"+f.name)
            print("deleting file at dir ", directory+"/"+f.name)

def get_data(): 
    inp_data = datasets.ImageFolder('./Data', transform = transforms.Compose([transforms.Resize(256), 
                                                                              transforms.CenterCrop(256),
                                                                              transforms.ToTensor(),]))
    
    inp = torch.utils.data.DataLoader(inp_data, batch_size = 10, shuffle = True)

    return inp


if __name__ == "__main__":
    remove_irrelivent("./Data")
