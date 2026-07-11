# This esimator is statless and does not to be fitted.
# However , we recommend to call fit_transform instead of transform, as parameter validation is only performed in fit

from sklearn.datasets import load_sample_images
from sklearn.feature_extraction import image

# Use the array data from the second image on this dataset

X = load_sample_images().images[1]
X = X[None, ...]
# print(f"Image shape: {X.shape}")

pe = image.PatchExtractor(patch_size=(10,10))
pe_trans = pe.transform(X)
# print(f"Patches shape: {pe_trans.shape}")

X_reconstructed = image.reconstruct_from_patches_2d(pe_trans,X.shape[1:])
print(f"Reconstructed shape: {X_reconstructed.shape}")
