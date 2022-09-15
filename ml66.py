
import os,torch
from nilearn import plotting
from nilearn import datasets
atlas = datasets.fetch_atlas_msdl()


atlas_filename = "64/64/2mm/maps.nii.gz"
def whatCellType(input_size, hidden_size, dropout_rate):
   return         torch.nn.RNN(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
def f(path3):
    d1ata=[]

    for i in os.listdir(path3):
                #nilearn.datasets.fetch_development_fmri()
                import torch
                from nilearn import plotting
                from nilearn import datasets
                myatlas=atlas
            # myatlas.
                # Loading atlas image stored in 'maps'
                #atlas_filename = myatlas.fetch_atlas_msdl()
                # Loading atlas data stored in 'labels'
                labels = atlas['labels']
                from nilearn.maskers import NiftiMapsMasker
                masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                                        memory='nilearn_cache', verbose=5)

                time_series = masker.fit_transform(imgs=path3+"/"+i)
                try:
                    from sklearn.covariance import GraphicalLassoCV
                except ImportError:
                    # for Scitkit-Learn < v0.20.0
                    from sklearn.covariance import GraphLassoCV as GraphicalLassoCV

                estimator = GraphicalLassoCV()
                estimator.fit(time_series)
                d1ata.append(estimator.covariance_)
    # Display the covariance
    # The covariance can be foundestimator.covariance_
                
    return d1ata
import pandas as pd
print(pd.DataFrame(f("out")).to_html())
print(pd.DataFrame(f("control")).to_html())