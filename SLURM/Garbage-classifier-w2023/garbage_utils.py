import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import glob
from sklearn.model_selection import StratifiedShuffleSplit
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


# Function to get the statistics of a dataset
def get_dataset_stats(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in data_loader:
        data = data[0]  # Get the images to compute the stgatistics
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


class TorchVisionDataset(Dataset):
    def __init__(self, data_dic, transform=None):
        self.file_paths = data_dic["X"]
        self.labels = data_dic["Y"]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)
        return image, label


class GarbageModel(nn.Module):
    def __init__(self,  num_classes, input_shape, transfer=False):
        super().__init__()

        self.transfer = transfer
        self.num_classes = num_classes
        self.input_shape = input_shape

        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet18(pretrained=transfer)

        if self.transfer:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        n_features = self._get_conv_output(self.input_shape)
        self.classifier = nn.Linear(n_features, num_classes)

    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.feature_extractor(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # will be used during inference
    def forward(self, x):
       x = self.feature_extractor(x)
       x = x.view(x.size(0), -1)
       x = self.classifier(x)
       
       return x


def get_data_loaders(images_path, val_split, test_split, batch_size=32, verbose=True):
    """
    These function generates the data loaders for our problem. It assumes paths are
    defined by "/" and image files are jpg. Each subfolder in the images_path 
    represents a different class.

    Args:
        images_path (_type_): Path to folders containing images of each class.
        val_split (_type_): percentage of data to be used in the val set
        test_split (_type_): percentage of data to be used in the val set
        verbose (_type_): debug flag

    Returns:
        DataLoader: Train, validation and test data laoders.
    """

    # Listing the data
    images = glob.glob(images_path + "*/*.jpg")
    images = np.array(images)
    labels = np.array([f.split("/")[-2] for f in images])

    # Formatting the labs as ints
    classes = np.unique(labels).flatten()
    labels_int = np.zeros(labels.size, dtype=np.int64)

    # Convert string labels to integers
    for ii, jj in enumerate(classes):
        labels_int[labels == jj] = ii

    if verbose:
        print("Number of images in the dataset:", images.size)
        for ii, jj in enumerate(classes):
            print("Number of images in class ", jj,
                  ":", (labels_int == ii).sum())

    # Splitting the data in dev and test sets
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_split, random_state=10)
    sss.get_n_splits(images, labels_int)
    dev_index, test_index = next(sss.split(images, labels_int))

    dev_images = images[dev_index]
    dev_labels = labels_int[dev_index]

    test_images = images[test_index]
    test_labels = labels_int[test_index]

    # Splitting the data in train and val sets
    val_size = int(val_split*images.size)
    val_split = val_size/dev_images.size
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_split, random_state=10)
    sss2.get_n_splits(dev_images, dev_labels)
    train_index, val_index = next(sss2.split(dev_images, dev_labels))

    train_images = images[train_index]
    train_labels = labels_int[train_index]

    val_images = images[val_index]
    val_labels = labels_int[val_index]

    if verbose:
        print("Train set:", train_images.size)
        print("Val set:", val_images.size)
        print("Test set:", test_images.size)

    # Representing the sets as dictionaries
    train_set = {"X": train_images, "Y": train_labels}
    val_set = {"X": val_images, "Y": val_labels}
    test_set = {"X": test_images, "Y": test_labels}

    # Transforms
    torchvision_transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                                      transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                                      transforms.ToTensor()])

    # Datasets
    train_dataset_unorm = TorchVisionDataset(
        train_set, transform=torchvision_transform_train)

    # Get training set stats
    trainloader_unorm = torch.utils.data.DataLoader(
        train_dataset_unorm, batch_size=batch_size, shuffle=True, num_workers=0)
    mean_train, std_train = get_dataset_stats(trainloader_unorm)

    if verbose:
        print("Statistics of training set")
        print("Mean:", mean_train)
        print("Std:", std_train)

    torchvision_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                                transforms.ToTensor(), transforms.Normalize(mean=mean_train, std=std_train)])

    torchvision_transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                                     transforms.ToTensor(), transforms.Normalize(mean=mean_train, std=std_train)])

    # Get the train/val/test loaders
    train_dataset = TorchVisionDataset(
        train_set, transform=torchvision_transform)
    val_dataset = TorchVisionDataset(val_set, transform=torchvision_transform)
    test_dataset = TorchVisionDataset(
        test_set, transform=torchvision_transform_test)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0)

    return trainloader, valloader, testloader


def train_validate(net, trainloader, valloader, epochs, batch_size,
                   learning_rate, best_model_path, device, verbose):

    best_loss = 1e+20
    for epoch in range(epochs):  # loop over the dataset multiple times

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Loss function
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        # Training Loop
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end=' ')
        scheduler.step()

        val_loss = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
            print(f'val loss: {val_loss / i:.3f}')

            # Save best model
            if val_loss < best_loss:
                print("Saving model")
                torch.save(net.state_dict(), best_model_path)
                best_loss = val_loss

    print('Finished Training')


def test(net, testloader):

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the test images: {100 * correct / total} %')
