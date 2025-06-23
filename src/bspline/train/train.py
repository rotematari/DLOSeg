import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from bspline.loss.loss import total_arclength_loss,ChamferLoss,resample,RMSELoss
def run_epoch(model, loader, optimizer, device, train=True, out_dim=3, teacher_forcing_ratio=0.1,progress=0):
    mse_loss = nn.MSELoss()
    chamfer_loss = ChamferLoss()
    rmse_loss = RMSELoss()
    total_loss = 0.0
    model.train() if train else model.eval()

    with torch.set_grad_enabled(train):
        for mask, spline_gt in loader:
            mask, spline_gt = mask.to(device), spline_gt.to(device)
            # spline_gt_deriv = torch.zeros_like(spline_gt)
            # spline_gt_deriv[:,:spline_gt.shape[1]-1] = spline_gt[:, 1:] - spline_gt[:, :-1]  # Derivative of the spline points

            if train:
                optimizer.zero_grad()
            pts_pred = model(mask)
            # loss for n end points
            # n = 5
            # loss_start = mse(pts_pred[:, :n, :], spline_gt[:, :n, :])
            # loss_end = mse(pts_pred[:, -n:, :], spline_gt[:, -n:, :])
            # loss_pts = mse(pts_pred, spline_gt)

            # loss = chamfer_loss(pts_pred, spline_gt) \
            #         + 0.1 * total_arclength_loss(pts_pred, spline_gt) \
                    # + 0.2 * soft_dice_loss(pts_pred, spline_gt, res=64, sigma=0.01)
            dense_T = 1000  # Number of points to resample the spline
            gt_dense = resample(spline_gt, dense_T)
            loss =  rmse_loss(pts_pred, spline_gt)# + \
                    # 0.05*chamfer_loss(pts_pred, gt_dense) + \
                    # 0.01*total_arclength_loss(pts_pred, spline_gt)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * mask.size(0)

    return total_loss / len(loader.dataset)

def train(model, train_loader, val_loader, optimizer, device, num_epochs=10, out_dim=3,results_root="./results"):
    best_val_loss = float('inf')
    val_losses = []
    train_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()
        progress = epoch / num_epochs  # Decrease progress as epochs increase
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True, out_dim=out_dim,progress=progress)
        end_time = time.time()
        # calc inference time
        inference_time = (end_time - start_time)/ len(train_loader.dataset)
        print(f"Inference time per sample: {inference_time:.6f} seconds")
        train_losses.append(train_loss)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False, out_dim=out_dim)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model with validation loss:", best_val_loss)
            
        if epoch % 2 == 0:
            visualize_examples(model, val_loader.dataset, device, n_examples=4, out_dim=out_dim,results_root=results_root)
    print("Training complete. Best validation loss:", best_val_loss)
    
    return train_losses, val_losses
def visualize_examples(model, dataset, device, n_examples=4, out_dim=3, results_root="./results"):
    # clear past figures
    plt.close('all')
    
    model.eval()
    fig = plt.figure(figsize=(12, 3 * n_examples))
    for i in range(n_examples):
        mask, spline_gt = dataset[i]
        with torch.no_grad():

            pred = model(mask.unsqueeze(0).to(device))[0].cpu().numpy()

        gt = spline_gt.numpy()
        
        if out_dim == 3:
            ax = fig.add_subplot(n_examples, 2, 2 * i + 1, projection='3d')
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'g', label='GT')
            ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'r--', label='Pred')
            ax.set_title(f"3D Curve #{i}")
            ax.legend()

        ax2 = fig.add_subplot(n_examples//2, 2, i+1)
        ax2.plot(gt[:, 0], gt[:, 1], 'g', label='GT XY')
        ax2.plot(pred[:, 0], pred[:, 1], 'r--', label='Pred XY')
        ax2.set_aspect('equal', 'box')
        ax2.set_title(f"2D Projection #{i}")
        ax2.legend()
        
    plt.tight_layout()
    fig_name = f"{results_root}/examples.png"
    plt.savefig(fig_name)
    
if __name__ == "__main__":
    
    pass
    # unit test for visualize_examples