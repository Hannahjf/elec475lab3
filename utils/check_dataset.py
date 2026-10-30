
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from utils.dataset import PetNoseDataset, denormalize


def main():
    # Check every sample in both splits
    for split in ["train", "test"]:
        dataset = PetNoseDataset(split=split)
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )

        checked = 0

        for images, labels, orig_sizes in loader:
            # Verify image dimensions
            assert images.shape[1:] == (3, 227, 227)

            # Verify all nose coordinates are inside the image
            assert torch.isfinite(labels).all()
            assert ((labels >= 0) & (labels < 227)).all()

            checked += images.shape[0]

        print(f"{split}: verified {checked} samples")

    # Plot eight training images
    dataset = PetNoseDataset(split="train")

    fig, axes = plt.subplots(2, 4, figsize=(12, 7))

    for i, ax in enumerate(axes.flat):
        image, label, orig_size = dataset[i]

        # Undo ImageNet normalization for display
        image = denormalize(image)

        # Change (C, H, W) to (H, W, C)
        image = image.permute(1, 2, 0)

        ax.imshow(image)
        ax.scatter(label[0].item(), label[1].item(),
                   c="red", s=45)

        ax.set_title(f"Sample {i}")
        ax.axis("off")

    plt.tight_layout()

    # Save visualization
    Path("plots").mkdir(exist_ok=True)
    plt.savefig("plots/dataset_check.png", dpi=150)

    print("Saved plots/dataset_check.png")
    plt.close()


if __name__ == "__main__":
    main()
