# libraries
import numpy as np

# import embeds
rt_bg = np.load('embeds_vae_receptor_fromReceptorTrain.npz', allow_pickle = True)['arr_0'] # embeds from the VAE trained on the split receptor train data
mb_bg = np.load('embeds_vae_receptor_fromBGMimic.npz', allow_pickle = True)['arr_0'] # mimic bg images that Bastian sent

print(rt_bg[0], mb_bg[0])
print(mb_bg[0] - rt_bg[0])
