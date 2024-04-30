import torch
from torch.utils.data import Dataset,DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

class PairedImageDataset(Dataset):
    def __init__(self, img_dir1, img_dir2, transform=None):
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.transform = transform
        
        self.img_filenames1 = sorted(os.listdir(img_dir1))
        self.img_filenames2 = sorted(os.listdir(img_dir2))

    
    def __len__(self):
        return len(self.img_filenames1)
    
    def __getitem__(self, idx):
        img_path1 = os.path.join(self.img_dir1, self.img_filenames1[idx])
        img_path2 = os.path.join(self.img_dir2, self.img_filenames2[idx])
        
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        # Return the pair of images
        return image1, image2
    
# ground_truth_set = torchvision.datasets.ImageFolder("ImageNet50/train", transform)
# corrupted_set = torchvision.datasets.ImageFolder("ImageNet50/train_motion_blur4", transform)
train_set =  PairedImageDataset("ImageNet50/train/n01532829","ImageNet50/train_gaussian_noise4/n01532829", transform)
test_set =  PairedImageDataset("ImageNet50/test/n01532829","ImageNet50/test_gaussian_noise4/n01532829", transform)
# train_set =  PairedImageDataset("ImageNet50/train2","ImageNet50/train2_blur", transform)


combined_loader = DataLoader(train_set,32 ,shuffle=True, num_workers= 8)
combined_loader2 = DataLoader(test_set,32 ,shuffle=True, num_workers= 8)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32 ,3,2),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(193600, 128)

        )
        self.transition_layer = nn.Linear(128, 193600)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,2,2,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,3,2,output_padding=1),
            nn.ReLU()

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.transition_layer(x)
        x = torch.reshape(x,(x.shape[0],64,55,55))
        x = self.decoder(x)
        return x

class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12288,6144),
            nn.ReLU(),
            nn.Linear(6144, 3072),
            nn.ReLU(),
            nn.Linear(3072, 1536),
            nn.ReLU(),
            nn.Linear(1536, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),

          
          
        )
        self.decoder = nn.Sequential(
            nn.Linear(256,768),
            nn.ReLU(),
            nn.Linear(768, 1536),
            nn.ReLU(),
            nn.Linear(1536, 3072),
            nn.ReLU(),
            nn.Linear(3072, 6144),
            nn.ReLU(),
            nn.Linear(6144, 12288),
            nn.ReLU(),
            

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.reshape(x,(x.shape[0],3,64,64))
        return x

model = Autoencoder().to(device)

loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)
num_epoch = 500

model.train()
for epoch in range(num_epoch):
    total_loss_per_epoch = 0
    for gt, corrupted in combined_loader:
        # print(gt.shape)
        # print(corrupted.shape)
        ground_truth_image = gt.to(device)
        corrupted_image = corrupted.to(device)
        save_image(ground_truth_image[0],"gt_test.png")
        save_image(corrupted_image[0],"cor_test.png")

        output = model(corrupted_image)

        save_image(output[0],"de_test.png")


        loss = loss_f(output, ground_truth_image)   
        total_loss_per_epoch += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
    print(total_loss_per_epoch)
    
torch.save(model.state_dict(), 'model_state_dict.pth')

model.eval()
for epoch in range(1):
    total_loss_per_epoch = 0
    for gt, corrupted in combined_loader2:
        ground_truth_image = gt.to(device)
        corrupted_image = corrupted.to(device)
        save_image(ground_truth_image[0],"gt_test_test.png")
        save_image(corrupted_image[0],"cor_test_test.png")

        output = model(corrupted_image)

        save_image(output[0],"de_test_test.png")
        loss = loss_f(output, ground_truth_image)   
        total_loss_per_epoch += loss.item()
      
        # print(loss)
    print(total_loss_per_epoch)

