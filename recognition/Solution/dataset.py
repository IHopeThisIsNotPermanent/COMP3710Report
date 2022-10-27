#dataset.py
import torch, os
from torchvision import datasets, transforms

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

#dataset = datasets.ImageFolder('path', transform=transform)


if __name__ == "__main__":
    remove_irrelivent("./Data")