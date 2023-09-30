import torch
import numpy as np
from tqdm import tqdm
import csv
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from skimage import io



data_file = "sim Data/driving_log.csv"
DATA = [0,0,0] 
prepare_batch_X, prepare_batch_y= [],[]
use_images = 1275
number_to_split_datasets = 1000



# Create dataset class
class Data(Dataset):
    def __init__(self, images, targets):
        self.x = images
        self.y = targets
        self.len = len(self.x)
    def __getitem__(self,index):
           return self.x[index], self.y[index]
    def __len__(self):
        return self.len

# Read CSV File and save the data
with open(f'{data_file}', 'r') as file:
    reader = csv.reader(file)
    # Shuffle the data
    shuffler = []
    i = 0
    for row in reader:
        shuffler.append(row)

        if(i > use_images):
            break
        i += 1
    random.shuffle(shuffler)



    # One by one append the shuffled data to our data array
    for row in shuffler:
        if DATA[0] == 0:
            DATA[0],DATA[1],DATA[2]  = [row[0]],[row[3]], [row[4]]
            init = False
            continue
        DATA[0].append(row[0])
        DATA[1].append(row[3])
        DATA[2].append(row[4])
       


for i in tqdm(range(0, len(DATA[0]), 1)):
   
    path_to_image = DATA[0][i]
  
    img = io.imread(path_to_image)
   
    # Prepare images
    prepare_batch_X.append(np.array(img))
    # Prepare steering angle and throttle data
    prepare_batch_y.append(np.array([  float(DATA[1][i]),float(DATA[2][i])  ]))
   
 





#Format data for training
# format image data /255.0 so our pixels are in range (0-1)
prepare_batch_X = torch.tensor(prepare_batch_X, dtype=torch.float).view(-1,3,160,320)/255.0
prepare_batch_y = torch.tensor(prepare_batch_y, dtype=torch.float).view(-1, 2)




print(f"prepare_batch_X.shape: {prepare_batch_X.shape}, prepare_batch_y.shape: {prepare_batch_y.shape}")

# Split data for training/validation
X_train = prepare_batch_X[:number_to_split_datasets]
y_train = prepare_batch_y[:number_to_split_datasets]

X_test = prepare_batch_X[number_to_split_datasets:]
y_test = prepare_batch_y[number_to_split_datasets:]




# # Create the datasets which will be use for training validation *this data will be exported to train file
dataset_train = Data(X_train, y_train)
dataset_test = Data(X_test, y_test)



train_loader = DataLoader(dataset = dataset_train, batch_size = 32)
test_loader = DataLoader(dataset = dataset_test, batch_size = 32)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






#Create the CNN
class DriverNet(nn.Module):

  def __init__(self):
        super(DriverNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*13*33, out_features=100),
            nn.ELU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=100, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=2)
        )
        

  def forward(self, x):
      x = x.view(x.size(0), 3, 160, 320)
      output = self.conv_layers(x)
      output = output.view(output.size(0), -1)
      output = self.linear_layers(output)
      return output
    



model =DriverNet().to(device)

#Model parameters
lr = 0.0005
epochs = 30
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()


#Train the data 
for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
        

            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()

            optimizer.step()

        loss_valid = 0.0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
        

            yhat = model(x)
            loss = criterion(yhat, y)

           
            loss_valid += loss

      
        #calculating the loss for each epoch
        print(f"epoch: {epoch+1}, validation loss: {loss_valid}")
    

#saving the model
torch.save(model.state_dict(), "TrainedModel.pth")


