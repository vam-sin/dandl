# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
sns.set_theme()
sns.set(font="Verdana")

# import data annots
ds_patches = pd.read_csv('gen_files/patches_csv_bg_addSpots_4__ONLY1.csv')
patch_annots = list(ds_patches["patch_annotation"])
colors_annots = []

for i in range(len(patch_annots)):
    if patch_annots[i] == 0: # no spots
        colors_annots.append('#e84118')
    else:
        colors_annots.append('#27ae60')
# print(colors_annots)
print(len(patch_annots))

# import data embeddings
embeds_vae = np.load('gen_files/embeds_vae_BGAddSpots_run1__ONLY1.npz', allow_pickle = True)['arr_0']
embeds_vae = np.squeeze(embeds_vae, axis=1)
print(embeds_vae.shape)

# plot a UMAP of the embeds
reducer = umap.UMAP()
low_dim_embeds = reducer.fit_transform(embeds_vae)
print(low_dim_embeds.shape)
# low_dim_embeds = embeds_vae

# plot
plt.scatter(
    low_dim_embeds[:, 0],
    low_dim_embeds[:, 1],
    c=colors_annots)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP of VAE-BGSynth_run1 on Receptor Test', fontsize=20)
plt.show()
