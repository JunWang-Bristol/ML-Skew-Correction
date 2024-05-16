# %%
import os
from PIL import Image
import torch, gc
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, Subset
import matplotlib.pyplot as plt
import numpy as np
import random

# #relaesa VRAM
# gc.collect()
# torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

# %%

# preprocessing, converting to grayscale, resize to 256 * 256 and normalizing the images, 
# and keep the channel dimension as the first dimension of the tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class CustomDataset(Dataset):
    def __init__(self, root_folder):
        self.image_paths = []  # complete path of all image files
        self.labels = []  # store the label information of all images
        self.case_ids = []  # store the operation condition number corresponding to each image
        
        # traverse all subfolders under the root folder
        for folder_name in sorted(os.listdir(root_folder), key=lambda x: int(x.split('_')[0])):
            folder_path = os.path.join(root_folder, folder_name)
            # extract physical properties from the folder name
            _, hmax, hmin, bmax, bmin = folder_name.split('_')
            hmax = float(hmax[4:])
            hmin = float(hmin[4:])
            bmax = float(bmax[4:])
            bmin = float(bmin[4:])
            
            case_id = folder_name.split('_')[0]  # extract the operation condition number
            # traverse all image files in the folder
            for image_file in sorted(os.listdir(folder_path), key=lambda x: int(x[:-4])):
                # create the complete file path and add it to the list
                self.image_paths.append(os.path.join(folder_path, image_file))
                
                # 解析offset_label
                offset_label = int(image_file[:-4]) - 20

                # add the label information as a tuple to the label list
                self.labels.append((offset_label, hmax, hmin, bmax, bmin))
                self.case_ids.append(case_id)  # store the operation condition number

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # get the image tensor
        image_path = self.image_paths[idx] # get the image path
        image = Image.open(image_path).convert('L') # convert to grayscale
        image_tensor = transform(image)
    
        # get the label information
        offset_label, hmax, hmin, bmax, bmin = self.labels[idx]

        case_id = self.case_ids[idx]  # get the operation condition number

        # hmax, hmin, bmax, bmin as extra inputs tensor
        extra_inputs = torch.tensor([hmax, hmin, bmax, bmin], dtype=torch.float32)

        # return the image tensor, label tensor, extra inputs tensor, and case_id
        return image_tensor, offset_label, extra_inputs, case_id

# set your dataset folder path
dataset_folder = './data_truncated'
dataset = CustomDataset(dataset_folder)

# %% randomly pick 1000 operation points as the test set and the rest as the training set
# set the random seed
np.random.seed(42) # to ensure reproducibility

# create an array of indices
indices = np.arange(1197)

# shuffle the indices
np.random.shuffle(indices)

# pick 1000 operation points as the test set
test_indices = indices[:1000]

# the rest as the training set
train_indices = indices[1000:]

# transform the indices to the actual data points
# 41 data points for each operation point
test_indices = np.hstack([np.arange(i * 41, i * 41 + 41) for i in test_indices])
train_indices = np.hstack([np.arange(i * 41, i * 41 + 41) for i in train_indices])

# create the training and test datasets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=197, shuffle=False)

# %% ##############################################
# define a CNN network 
from nn import CNN

model = CNN().to(device)
model_path_load = './model_checkpoint_org_local.pth'
model.load_state_dict(torch.load(model_path_load, map_location=torch.device('cpu')))

# %% ############################################## Preduction
# Create a folder to store the prediction images
prediction_folder = './prediction_global+local'
if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)
    
# Set the model to evaluation mode
model.eval()

# randomly select 100 samples from the test set
indices = random.sample(range(len(test_dataset)), 100)

# load test samples
test_samples = [test_dataset[i] for i in indices]

# Initialize a list to store prediction errors
prediction_errors = []

# predict the offset time for each sample and visualize the results
for idx, (image_tensor, offset_label, extra_inputs, case_id) in enumerate(test_samples):
    image_tensor_unsqueeze = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
    extra_inputs = extra_inputs.unsqueeze(0).to(device)  # Prepare model input

    prediction = model(image_tensor_unsqueeze, extra_inputs)  # Perform prediction


    # uncomment the following line if precise prediction is needed
    # predicted_offset = prediction.item()  # Get the predicted value


    predicted_offset = round(prediction.item()) # Get the predicted value and round to the nearest integer

    # Calculate prediction error
    error = abs(predicted_offset - offset_label)
    prediction_errors.append(error)  # Append error to the list

    # Extract specific max and min values from extra_inputs
    hmax, hmin, bmax, bmin = extra_inputs[0].cpu().numpy()  # Move to CPU and convert to numpy

     # Visualize the image and prediction results
    plt.figure(figsize=(6, 6))
    plt.imshow(image_tensor.squeeze(0).numpy(), cmap='gray')  # Display the image
    plt.title(f"Case: {case_id}, Predicted Time: {predicted_offset:.2f}, Actual Time: {offset_label}\n"
              f"Hmax: {hmax:.4f}, Hmin: {hmin:.4f}, Bmax: {bmax:.4f}, Bmin: {bmin:.4f}")
    plt.axis('off')  # axis off

    # Save the image
    plt.savefig(os.path.join(prediction_folder, f'prediction_{case_id}_{idx}.png'))
    plt.close()  # Close the plot

# Calculate average prediction error
average_error = sum(prediction_errors) / len(prediction_errors)

# Write the average error and individual predictions to a text file
with open(os.path.join(prediction_folder, 'predictions.txt'), 'w') as f:
    for error in prediction_errors:
        f.write(f"{error}\n")
    f.write(f"Average Error: {average_error:.4f}\n")  # Write average error at the end of the file

# %%
