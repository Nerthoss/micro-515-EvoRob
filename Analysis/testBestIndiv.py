import numpy as np

# Verify the saved genotype is actually individual 75 from gen 98
x_saved = np.load("results/test02_200gen/x_best.npy")
x_gen98 = np.load("results/test02_200gen/98/x.npy")[75]

print(f"Files match: {np.allclose(x_saved, x_gen98)}")  # must be True
print(f"Shape: {x_saved.shape}")  # must be (252,)