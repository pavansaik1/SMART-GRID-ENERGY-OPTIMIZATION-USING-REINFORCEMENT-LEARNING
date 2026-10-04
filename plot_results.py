import matplotlib.pyplot as plt

strategies = ["A2C", "Random", "Rule-based"]
costs = [158.03, 224.13, 159.17]

plt.figure(figsize=(8, 5))
bars = plt.bar(strategies, costs)

plt.title("Average Grid Electricity Cost")
plt.ylabel("Average cost (project units)")
plt.xlabel("Control strategy")

for bar, cost in zip(bars, costs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{cost:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("results/cost_comparison.png", dpi=300)
plt.show()