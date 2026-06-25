import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class PlotManager:
    @staticmethod
    def _get_output_dir(title):
        title_lower = title.lower()
        if "alpha" in title_lower:
            return Path("results/alpha")
        if "epsilon" in title_lower or "exploration" in title_lower:
            return Path("results/epsilon")
        if "baseline" in title_lower:
            return Path("results/baseline")
        return Path("results/evaluation")

    @staticmethod
    def _save_plot(fig, title, output_path=None):
        if output_path is None:
            output_dir = PlotManager._get_output_dir(title)
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
            output_path = output_dir / f"{safe_name}.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {output_path}")
        return output_path

    @staticmethod
    def plot_rewards(rewards, title="Reward per Episode", output_path=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rewards)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True)
        fig.tight_layout()
        return PlotManager._save_plot(fig, title, output_path)

    @staticmethod
    def plot_steps(steps, title="Steps per Episode", output_path=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, color="orange")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.grid(True)
        fig.tight_layout()
        return PlotManager._save_plot(fig, title, output_path)

    @staticmethod
    def plot_moving_average(rewards, window=100, title="Moving Average Reward", output_path=None):
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(moving_avg, color="green")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Reward")
        ax.grid(True)
        fig.tight_layout()
        return PlotManager._save_plot(fig, title, output_path)
