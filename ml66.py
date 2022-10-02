
from urllib import request
import bottle
from bottle import request
import os
import sys
import pandas as pd

# routes contains the HTTP handlers for our server and must be imported.
app=bottle
if '--debug' in sys.argv[1:] or 'SERVER_DEBUG' in os.environ:
    # Debug mode will enable more verbose output in the console window.
    # It must be set at the beginning of the script.
    bottle.debug(True)

import os,torch
from nilearn import plotting
from nilearn import datasets
atlas = datasets.fetch_atlas_msdl()


atlas_filename = "64/64/2mm/maps.nii.gz"
def whatCellType(input_size, hidden_size, dropout_rate):
   return         torch.nn.RNN(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
def f(path3,size):
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
                if len(d1ata)>size:
                    return d1ata
    # Display the covariance
    # The covariance can be foundestimator.covariance_
                
    return d1ata

@bottle.route("/c")
def controlgru():
    t=torch.tensor(f("control",2))
    a=0
    b=t

    g=torch.nn.AdaptiveMaxPool2d((256,64))
    b=g(t)
    
    relu=torch.nn.ReLU()
    b=relu(b)
    re=torch.nn.AdaptiveMaxPool2d((2,2))
    b=re(b)
    
    #b=list(b)
    #b1=torch.tensor(b)

    #s=torch.nn.Softmax(b)
    #b=s(b)
    print(b)
    a=a+1

    return str(b)

@bottle.route("/s")
def schizogro():
    
    f("out",2)
    import numpy as np
    t=torch.tensor(f("out",3))
    a=0
    b=t

    g=torch.nn.AdaptiveMaxPool2d((256,64))
    b=g(t)
    
    relu=torch.nn.ReLU()
    b=relu(b)
    re=torch.nn.AdaptiveMaxPool2d((2,2))
    b=re(b)
    
    #b=list(b)
    #b1=torch.tensor(b)

    #s=torch.nn.Softmax(b)
    #b=s(b)
    print(b)
    a=a+1

    return str(b)

    #for aai in (f("out")[0:20]):
     #   torch.tensor(aai)
    #j=torch.nn.MaxPool3d(3,2)
    import torch.optim as optim
  
    criterion = torch.nn.HingeEmbeddingLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    return (pd.DataFrame(Net(t)).to_html(escape=False))
#print(pd.DataFrame(f("out")).to_csv())
#print(pd.DataFrame(f("control")).to_csv())
@bottle.route("/")
def a():
    return "<iframe src='/c'><iframe src='/s'>"
def wsgi_app():
    """Returns the application to make available through wfastcgi. This is used
    when the site is published to Microsoft Azure."""
    return bottle.default_app()

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    STATIC_ROOT = os.path.join(PROJECT_ROOT, 'static').replace('\\', '/')
    HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
    
    PORT = 5310

    @bottle.route('/static/<filepath:path>')
    def server_static(filepath):
        """Handler for static files, used with the development server.
        When running under a production server such as IIS or Apache,
        the server should be configured to serve the static files."""
        return bottle.static_file(filepath, root=STATIC_ROOT)

    # Starts a local test server.
    bottle.run(server='wsgiref', host=HOST, port=PORT)
