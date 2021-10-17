from PIL import Image
from torchvision import transforms
import glob
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DatasetManager:
    def __init__(self, data_dir, train_transform=None):
        self.filenames, self.labels = self.get_data(data_dir)
        # Test dataset
        if train_transform:
            self.train_transform = train_transform
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                # transforms.RandomErasing()
            ])

        self.valid_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_data(self, data_dir):
        filenames = []
        labels = []
        for filename in glob.glob(os.path.join(data_dir, '**/**')):
            filenames.append(filename)
            label = os.path.split(filename)[-2]
            labels.append(label)
        return filenames, labels

    def get_train_dataset(self, X, y):
        return CustomDataset(X, y, self.train_transform)

    def get_valid_dataset(self, X, y):
        return CustomDataset(X, y, self.valid_transform)

    def split(self, size=0.2):
        X_train, X_valid, y_train, y_valid = train_test_split(self.filenames, self.labels, test_size=size,
                                                              random_state=0)
        return self.get_train_dataset(X_train, y_train), self.get_valid_dataset(X_valid, y_valid)


# Define dataset here
class CustomDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        super(CustomDataset, self).__init__()
        self.filenames, self.labels = filenames, labels
        self.label_mapping = self.get_label_mapping()
        self.transform = transform

    def get_label_mapping(self):
        label_mapping = {}
        for label in set(self.labels):
            label_mapping[label] = len(label_mapping)
        return label_mapping

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        input_fn = self.filenames[index]
        label = self.labels[index]
        image = Image.open(input_fn)
        image = self.transform(image)
        # one_hot_label = torch.zeros(len(self.label_mapping))
        # one_hot_label[self.label_mapping[label]] = 1
        return image, self.label_mapping[label]