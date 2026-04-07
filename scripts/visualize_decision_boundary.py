import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_decision_boundary(output_dir: str):
    """
    Visualize the T_local vs T_remote decision boundary (M.4).
    T_local = (N * d) / (P * FLOPS)
    T_remote = Lat_network + O(log k * d)
    """
    N = np.linspace(100, 150000, 1000)
    d = 384  # MiniLM
    P = 12  # Parallelization factor (typical modern CPU)
    FLOPS = 2.5e9  # Estimated effective FLOPS for dot products

    # Decisions based on benchmarks
    T_local = (N * d) / (P * 2e6)  # Adjusted for realistic processing speed per element
    T_remote = np.full_like(N, 60.0)  # Estimated 60ms network penalty (Redis/Search)

    plt.figure(figsize=(10, 6))
    plt.plot(
        N,
        T_local,
        label="Local Parallel $\mathcal{O}(N)$ (Proposed)",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        N,
        T_remote,
        "--",
        label="Remote Indexed Search (RedisSearch)",
        color="red",
        linewidth=2,
    )

    # Highlight Decision Boundary
    crossover = N[np.argmin(np.abs(T_local - T_remote))]
    plt.axvline(
        x=crossover,
        color="green",
        linestyle=":",
        label=f"Decision Boundary (N ≈ {int(crossover)})",
    )
    plt.fill_between(
        N,
        0,
        150,
        where=(N < crossover),
        color="blue",
        alpha=0.1,
        label="Local Efficiency Zone",
    )

    plt.title(
        "Search Decision Boundary: Local $\mathcal{O}(N)$ vs. Remote Indexed",
        fontsize=14,
    )
    plt.xlabel("Cache Size (Number of Entries N)", fontsize=12)
    plt.ylabel("Lookup Latency (ms)", fontsize=12)
    plt.ylim(0, 150)
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = os.path.join(output_dir, "decision_boundary.png")
    plt.savefig(output_path, dpi=300)
    print(f"Decision boundary visualization saved to {output_path}")


if __name__ == "__main__":
    visualize_decision_boundary("results")
