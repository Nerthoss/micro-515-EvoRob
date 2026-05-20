import numpy as np
import matplotlib.pyplot as plt
import glob

best_min = -np.inf
best_gen = None
best_idx = None

for path in sorted(glob.glob("results/test02_200gen/*/f.npy")):
    f = np.load(path)
    min_fitness = f.min(axis=1)  # worst terrain per individual
    idx = min_fitness.argmax()   # individual with best worst-case
    if min_fitness[idx] > best_min:
        best_min = min_fitness[idx]
        best_gen = path
        best_idx = idx

print(f"Best generalist at: {best_gen}, individual {best_idx}")
print(f"Worst-case fitness: {best_min:.2f}")

f_all = np.load("results/test02_200gen/198/f.npy")   # shape (pop, 3)
x_all = np.load("results/test02_200gen/198/x.npy")   # shape (pop, n_params)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(f_all[:, 0], f_all[:, 1], f_all[:, 2])
ax.set_xlabel("Flat"); ax.set_ylabel("Ice"); ax.set_zlabel("Hill")
plt.show()
