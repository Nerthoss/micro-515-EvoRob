import numpy as np
import matplotlib.pyplot as plt
import glob

generations = []
mean_fitnesses = []
best_fitnesses = []

for path in sorted(glob.glob("results/smoke_test12/*/f.npy")):
    f = np.load(path)
    gen = int(path.split("/")[-2])
    generations.append(gen)
    mean_fitnesses.append(f.mean(axis=0))  # mean per terrain
    best_fitnesses.append(f.max(axis=0))   # best per terrain

mean_fitnesses = np.array(mean_fitnesses)
best_fitnesses = np.array(best_fitnesses)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
terrains = ["Flat", "Ice", "Hill"]
for i, (ax, terrain) in enumerate(zip(axes, terrains)):
    ax.plot(generations, mean_fitnesses[:, i], label="Mean")
    ax.plot(generations, best_fitnesses[:, i], label="Best")
    ax.set_title(terrain)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
plt.tight_layout()
plt.savefig("convergence.png")
plt.show()