import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to CSV file with columns: step,total_steps,loss")
    parser.add_argument("--num_images", type=int, required=True,
                        help="Number of training images in the dataset")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Per-device train batch size used in training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps used in training")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of processes/GPUs used by accelerate")
    parser.add_argument("--smooth_window", type=int, default=50,
                        help="Rolling window size for smoothing")
    parser.add_argument("--output_png", type=str, default="loss_vs_epoch.png",
                        help="Output figure filename")
    args = parser.parse_args()

    # effective batch size across all devices
    effective_batch_size = (
        args.train_batch_size
        * args.gradient_accumulation_steps
        * args.world_size
    )

    steps_per_epoch = math.ceil(args.num_images / effective_batch_size)

    df = pd.read_csv(args.csv_file)

    required_cols = {"step", "loss"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}, got {df.columns.tolist()}")

    df = df.sort_values("step").reset_index(drop=True)
    df["epoch"] = df["step"] / steps_per_epoch
    df["loss_smooth"] = df["loss"].rolling(
        window=args.smooth_window, min_periods=1
    ).mean()

    print(f"Num images: {args.num_images}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Estimated steps per epoch: {steps_per_epoch}")
    print(f"Estimated total epochs trained: {df['epoch'].iloc[-1]:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["loss"], alpha=0.35, label="Raw loss")
    plt.plot(df["epoch"], df["loss_smooth"], linewidth=2, label=f"Smoothed loss (window={args.smooth_window})")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=200)
    print(f"Saved plot to {args.output_png}")

    # also save epoch-converted csv
    out_csv = args.output_png.replace(".png", ".csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved epoch-converted data to {out_csv}")


if __name__ == "__main__":
    main()