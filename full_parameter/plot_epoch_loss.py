import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True,
                        help="CSV file with columns: step,total_steps,loss")
    parser.add_argument("--num_images", type=int, required=True,
                        help="Number of training images")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs/processes used by accelerate")
    parser.add_argument("--output_png", type=str, default="epoch_loss.png")
    parser.add_argument("--output_csv", type=str, default="epoch_loss.csv")
    args = parser.parse_args()

    effective_batch_size = (
        args.train_batch_size
        * args.gradient_accumulation_steps
        * args.world_size
    )

    steps_per_epoch = math.ceil(args.num_images / effective_batch_size)

    df = pd.read_csv(args.csv_file)
    if "step" not in df.columns or "loss" not in df.columns:
        raise ValueError("CSV must contain 'step' and 'loss' columns.")

    df = df.sort_values("step").reset_index(drop=True)

    # epoch index starting from 1
    df["epoch"] = ((df["step"] - 1) // steps_per_epoch) + 1

    # mean loss for each epoch
    epoch_df = df.groupby("epoch", as_index=False)["loss"].mean()
    epoch_df.rename(columns={"loss": "training_loss"}, inplace=True)

    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total epochs: {epoch_df['epoch'].max()}")

    epoch_df.to_csv(args.output_csv, index=False)
    print(f"Saved epoch loss data to {args.output_csv}")

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_df["epoch"], epoch_df["training_loss"], marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss over Epochs")
    plt.xticks(epoch_df["epoch"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=200)
    print(f"Saved plot to {args.output_png}")


if __name__ == "__main__":
    main()