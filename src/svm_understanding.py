# libraries
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

# import data
ds_patches = pd.read_csv('patches_csv_receptor.csv')
patch_annots = list(ds_patches["patch_annotation"])
print(len(patch_annots))

# import data embeddings
embeds_vae = np.load('embeds_vae_receptor_fromReceptorTrain_run7.npz', allow_pickle = True)['arr_0']
embeds_vae = np.squeeze(embeds_vae, axis=1)
print(embeds_vae.shape)

# split data
X_train, X_test, y_train, y_test = train_test_split(
                                    embeds_vae, patch_annots, test_size = 0.2, random_state=42, shuffle=True)

# build the model
clf = svm.SVC()

print("Training")
clf.fit(X_train, y_train)
print("Testing")
print(clf.score(X_test, y_test))
