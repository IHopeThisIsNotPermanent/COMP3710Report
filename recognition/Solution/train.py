#train.py
import torch
from modules import UNET
from dataset import get_data

class trainer:

    def __init__(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.training_loader = get_data()
        
        self.model = UNET()
        
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr = 0.001, momentum = 0.9)
    
    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        
        for i, data in enumerate(self.training_loader):
            inputs, labels = data
            
            self.optimiser.zero_grad()
            
            outputs = self.model(inputs)
            
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            
            self.optimiser.step()
            
            running_loss == loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print(" back {} loss: {}".format(i+1, last_loss))
                running_loss = 0.
        
        return last_loss
    
if __name__ == "__main__":
    Train = trainer()
    