"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

# TODO: Choose from either model and uncomment that line
# class KeypointModel(nn.Module):
class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams, train_set = None, val_set = None):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        # self.hparams = hparams
        self.train_set = train_set
        self.val_set = val_set
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 7),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(20 * 20 * 20, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 30)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x
    
    def general_step(self, batch, batch_idx, mode):
        images = batch['image']
        gt_keypoints = batch['keypoints']

        # forward pass
        predicted_keypoints = self.forward(images).view(self.hparams['batch_size'], 15, 2)

        # loss
        loss = nn.MSELoss()(predicted_keypoints, gt_keypoints)

        return loss / images.size(0)

    def training_step(self, batch, batch_idx):
        loss  = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss  = self.general_step(batch, batch_idx, "val")
        return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'], drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'],  drop_last=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
