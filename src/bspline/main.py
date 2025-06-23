from bspline.datasets.datasets import SplinePointDataset
from bspline.models.models import SplinePointTransformer ,SimpleSplineNet,ConvNeXtSplineNet
from bspline.train.train import train , visualize_examples
import torch
from torch.utils.data import  DataLoader, random_split
import matplotlib.pyplot as plt

def main(dataset_path,
         results_root="/home/admina/segmetation/DLOSeg/src/bspline/results",
         img_size=(256, 256),
         num_pts=200,
         batch_size=32,
         
         epochs=10,
         learning_rate=1e-3,
         out_dim=2,
         val_size=0.1):
    
    EMBEDDING_DIM = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # load the dataset
    dataset = SplinePointDataset(root=dataset_path, img_size=img_size, num_pts=num_pts)
    val_size = int(val_size * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # initialize the model
    model = ConvNeXtSplineNet(
        num_spline_points=num_pts,
        out_dim=out_dim,
        freeze_backbone_at=None,  # freeze the backbone
        
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    train_losses, val_losses = train(model, train_loader, val_loader,
                                     optimizer, device, num_epochs=epochs, 
                                     out_dim=out_dim, results_root=results_root)

    # visualize some examples
    visualize_examples(model, val_loader.dataset, device, n_examples=4, out_dim=out_dim, results_root=results_root)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{results_root}/loss_plot.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a B-spline model.")
    parser.add_argument("--dataset_path", type=str, default = '/home/admina/segmetation/DLOSeg/dataset_3', help="Path to the dataset directory.")
    parser.add_argument("--img_size", type=int, nargs=2, default=(256, 256), help="Image size (height, width).")
    parser.add_argument("--num_pts", type=int, default=200, help="Number of points defining the spline.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--out_dim", type=int, default=2, help="Output dimension of the spline (2D or 3D).")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of dataset to use for validation.")
    parser.add_argument("--results_root", type=str, default="/home/admina/segmetation/DLOSeg/src/bspline/results", help="Root directory for saving results.")

    args = parser.parse_args()
    
    main(args.dataset_path,
         results_root=args.results_root,
         img_size=args.img_size,
         num_pts=args.num_pts,
         batch_size=args.batch_size,
         epochs=args.epochs,
         learning_rate=args.learning_rate,
         out_dim=args.out_dim)