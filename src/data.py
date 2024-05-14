from torchvision import transforms, datasets
import os
from torch.utils.data import DataLoader
from hyperparameters import Hyperparameters as hp

dataset_folder = "dataset_v3"

class DataHandler:
    @staticmethod
    def get_normalization_variables():
        train_path = os.path.join("..", "dataset_prep", dataset_folder, "train")
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load only training data to compute mean and std_dev
        train_data = datasets.ImageFolder(train_path, transform=train_transform)

        # Calculate mean and std_dev for each channel across all images
        mean_r = sum([data[0][0].mean() for data in train_data]) / len(train_data)
        mean_g = sum([data[0][1].mean() for data in train_data]) / len(train_data)
        mean_b = sum([data[0][2].mean() for data in train_data]) / len(train_data)

        std_r = sum([data[0][0].std() for data in train_data]) / len(train_data)
        std_g = sum([data[0][1].std() for data in train_data]) / len(train_data)
        std_b = sum([data[0][2].std() for data in train_data]) / len(train_data)

        # Combine means and std_dev for each channel into tuples
        mean = (mean_r, mean_g, mean_b)
        std_dev = (std_r, std_g, std_b)
        return {"std": std_dev, "mean": mean}

    @staticmethod
    def get_dataset():
        train_path = os.path.join("..", "dataset_prep", dataset_folder, "train")
        valid_path = os.path.join("..", "dataset_prep", dataset_folder, "test")
        test_path = os.path.join("..", "dataset_prep", dataset_folder, "valid")
        nv = DataHandler.get_normalization_variables()
        transform = transforms.Compose(
            [
                # transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(nv["mean"], nv["std"]),
            ]
        )

        # Load datasets
        train_data = datasets.ImageFolder(train_path, transform=transform)
        valid_data = datasets.ImageFolder(valid_path, transform=transform)
        test_data = datasets.ImageFolder(test_path, transform=transform)

        # print(train_data.imgs)

        # Data loaders
        train_loader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=hp.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader
