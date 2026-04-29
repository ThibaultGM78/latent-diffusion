import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(csv_path, output_dir="plots", show=False):
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    def save_plot(x, ys, labels, title, filename):
        plt.figure()
        for y, label in zip(ys, labels):
            plt.plot(x, y, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        if len(labels) > 1:
            plt.legend()
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        print(f"Saved: {filepath}")

    # --- VAE case ---
    if "recon_loss" in df.columns and "kl_loss" in df.columns:
        print("Detected VAE loss file")

        # Recompute total loss (safe)
        df["total_loss"] = df["recon_loss"] + 1e-6 * df["kl_loss"]

        save_plot(
            df["epoch"],
            [df["total_loss"]],
            ["Total Loss"],
            "VAE Total Loss",
            "vae_total_loss.png"
        )

        save_plot(
            df["epoch"],
            [df["recon_loss"]],
            ["Reconstruction Loss"],
            "VAE Reconstruction Loss (MSE)",
            "vae_recon_loss.png"
        )

        save_plot(
            df["epoch"],
            [df["kl_loss"]],
            ["KL Loss"],
            "VAE KL Divergence Loss",
            "vae_kl_loss.png"
        )

        save_plot(
            df["epoch"],
            [df["val_loss"]],
            ["Validation Loss"],
            "VAE Validation Loss",
            "vae_val_loss.png"
        )

    # --- Diffusion case ---
    elif "train_loss" in df.columns and "val_loss" in df.columns:
        print("Detected Diffusion loss file")

        save_plot(
            df["epoch"],
            [df["train_loss"], df["val_loss"]],
            ["Train Loss", "Validation Loss"],
            "Diffusion Model Loss",
            "diffusion_loss.png"
        )

    else:
        raise ValueError("Unknown CSV format. Expected VAE or Diffusion loss columns.")
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training losses from CSV")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing losses")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the plots")
    parser.add_argument("--show", action="store_true", help="Whether to display the plots after saving")
    args = parser.parse_args()

    plot_losses(args.csv_path, args.output_dir, args.show)