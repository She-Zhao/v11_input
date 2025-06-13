import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class MultiModalDataset(Dataset):
    def __init__(self, image_dirs, labels, transform=None):
        """
        :param image_dirs: List of directories, each containing images for one modality (N=6).
        :param labels: List of labels corresponding to each set of multi-modal images.
        :param transform: Optional transform to apply to each image.
        """
        self.image_dirs = image_dirs  # List of directories, one per modality
        self.labels = labels          # List of labels, one per sample (same for all modalities)
        self.transform = transform

        # Ensure all modalities have the same number of images
        num_images = [len(os.listdir(d)) for d in image_dirs]
        assert len(set(num_images)) == 1, "Each modality must have the same number of images."
        self.num_samples = num_images[0]  # Number of samples in each modality

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        images = []
        
        # Load each modality's image for the current index
        for modality_dir in self.image_dirs:
            image_path = os.path.join(modality_dir, f"{idx}.jpg")  # Modify as needed for your file format
            image = cv2.imread(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # Concatenate images along the channel dimension
        images_concat = torch.cat([torch.from_numpy(img).permute(2, 0, 1) for img in images], dim=0)

        # Get the label for this sample
        label = self.labels[idx]

        return images_concat, label

# Example usage:
image_dirs = ["modality1_images/", "modality2_images/", "modality3_images/",
              "modality4_images/", "modality5_images/", "modality6_images/"]
labels = [0, 1, 1, 0, 1, ...]  # List of labels for each sample

# Initialize dataset and dataloader
dataset = MultiModalDataset(image_dirs=image_dirs, labels=labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate through the dataloader
for images, label in dataloader:
    print(images.shape)  # Shape will be (batch_size, 6 * channels, height, width)
    print(label)         # Labels for each batch
