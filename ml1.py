from ast import Import
from email.mime import image
import os,PIL,nilearn
import torch.package as ew
import torch.types as TF
import torch.nn as nn
import torch._tensor as T5
import torch.nn.init as I
import torch.nn as NN
def whatCellType(input_size, hidden_size, dropout_rate):
   return         nn.RNN(input_size, hidden_size, dropout=dropout_rate, batch_first=False)


from pickletools import float8, uint8  
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import pandas as pd

inputs=[]
#nilearn.datasets.fetch_development_fmri()
            #import torch

for nii in os.listdir("c:/users/administrator/desktop/nii"):

    from nilearn import plotting
    from nilearn import datasets
    atlas = datasets.fetch_atlas_msdl()
    # Loading atlas image stored in 'maps'
    atlas_filename = "C:/Users/Administrator/Desktop/64/64/2mm/maps.nii.gz"
    # Loading atlas data stored in 'labels'
    labels = pd.read_csv("C:/Users/Administrator/Desktop/64/64/labels_64_dictionary.csv")
    a=labels.to_dict()
    b=a["Difumo_names"]
    from nilearn.maskers import NiftiMapsMasker
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                            memory='nilearn_cache', verbose=5)

    time_series = masker.fit_transform("c:/users/administrator/desktop/nii/"+nii)
    try:
        from sklearn.covariance import GraphicalLassoCV
    except ImportError:
        # for Scitkit-Learn < v0.20.0
        from sklearn.covariance import GraphLassoCV as GraphicalLassoCV

    estimator = GraphicalLassoCV()
    estimator.fit(time_series)
# Display the covariance
    nilearn.plotting.plot_img(estimator.covariance_, labels=list(a["Difumo_names"].values()),
                        figure=(9, 7), vmax=1, vmin=-1,
                        title='Covariance')# The covariance can be found at estimator.covariance_

# The covariance can be found at estimator.covariance_
    nilearn.plotting.plot_matrix(estimator.covariance_, labels=list(a["Difumo_names"].values()),
                        figure=(9, 7), vmax=1, vmin=-1,
                        title='Covariance')


for nii in os.listdir("c:/users/administrator/desktop/nii2"):

    from nilearn import plotting
    from nilearn import datasets
    atlas = datasets.fetch_atlas_msdl()
    # Loading atlas image stored in 'maps'
    atlas_filename = "C:/Users/Administrator/Desktop/64/64/2mm/maps.nii.gz"
    # Loading atlas data stored in 'labels'
    labels = pd.read_csv("C:/Users/Administrator/Desktop/64/64/labels_64_dictionary.csv")
    a=labels.to_dict()
    b=a["Difumo_names"]
    from nilearn.maskers import NiftiMapsMasker
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                            memory='nilearn_cache', verbose=5)

    time_series = masker.fit_transform("c:/users/administrator/desktop/nii/"+nii)
    try:
        from sklearn.covariance import GraphicalLassoCV
    except ImportError:
        # for Scitkit-Learn < v0.20.0
        from sklearn.covariance import GraphLassoCV as GraphicalLassoCV

    estimator = GraphicalLassoCV()
    estimator.fit(time_series)
# Display the covariance
    nilearn.plotting.plot_img(estimator.covariance_, labels=list(a["Difumo_names"].values()),
                        figure=(9, 7), vmax=1, vmin=-1,
                        title='Covariance')# The covariance can be found at estimator.covariance_
    nilearn.plotting.plot_matrix(estimator.covariance_, labels=list(a["Difumo_names"].values()),
                        figure=(9, 7), vmax=1, vmin=-1,
                        title='Covariance')


for i in os.listdir("c:/inetpub/wwwroot/out"):
    for j in os.listdir("c:/inetpub/wwwroot/out/"+i+"/"):
            #nilearn.datasets.fetch_development_fmri()
            import torch
            from nilearn import plotting
            from nilearn import datasets
            myatlas=atlas
           # myatlas.
            # Loading atlas image stored in 'maps'
            atlas_filename = atlas.fetch_atlas_msdl()
            # Loading atlas data stored in 'labels'
            labels = atlas['labels']
            from nilearn.maskers import NiftiMapsMasker
            masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                                    memory='nilearn_cache', verbose=5)

            time_series = masker.fit_transform(data.func[0],
                                            confounds=data.confounds)

# Display the covariance
# The covariance can be foundestimator.covariance_
            plotting.plot_matrix(estimator.covariance_, labels=labels,
                     figure=(9, 7), vmax=1, vmin=-1,
                     title='Covariance')
            import torchvision
            from torchvision.io import read_image
            import torchvision.transforms as T
            img = torchvision.io.read_image("c:/inetpub/wwwroot/out/"+i+"/"+j)
            out=None            
            weights =  torchvision.models.ResNet50_Weights.DEFAULT
            model =  torchvision.models.resnet50(weights=weights)
            model.eval()
            x=Image.open("c:/inetpub/wwwroot/out/"+i+"/"+j)
            
            # Step 2: Initialize the inference transforms
            preprocess = weights.transforms()
            
            # Step 3: Apply inference preprocessing transforms
            batch = preprocess(x.transform(size=(500,500),method=PIL.Image.Transform.AFFINE,data=x))

            # Step 4: Use the model and print the predicted category
            prediction = model(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = weights.meta["categories"][class_id]
            print(f"{category_name}: {100 * score:.1f}%")

            inputs.append(img)
            X=whatCellType(input_size=len(img),hidden_size=100,dropout_rate=0.003)
            

#np.array(input)
#pt_tensor_from_list =(np.array(hh,dtype=np.ndarray(hh).shape))

#print(X)
print(out)

