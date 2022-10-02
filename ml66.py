
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
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 6, 5)
        self.pool = torch.nn.MaxPool3d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 2)


    def forward(self, x):
        x = self.pool(torch.nn.ReLU6(self.conv1(x)))
        x = self.pool(torch.nn.ReLU6(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.nn.ReLU6(self.fc1(x))
        x = torch.nn.ReLU6(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

@bottle.route("/s")
def schizogro():
    
    f("out",2)
    import numpy as np
    t=torch.tensor(f("out",3))
    #for aai in (f("out")[0:20]):
     #   torch.tensor(aai)
    #j=torch.nn.MaxPool3d(3,2)
    import torch.optim as optim
  
    criterion = torch.nn.HingeEmbeddingLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times
        data=[]
        running_loss = 0.0
        for i in f("out",3-1):
            data.append(torch.tensor(i))
        for j1 in f("control",3-1):
            data.append(torch.tensor(j1))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net()
            loss = criterion(outputs, ["schizo","control"])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return (pd.DataFrame(Net(self,t)).to_html(escape=False))
#print(pd.DataFrame(f("out")).to_csv())
#print(pd.DataFrame(f("control")).to_csv())
def wsgi_app():
    """Returns the application to make available through wfastcgi. This is used
    when the site is published to Microsoft Azure."""
    return bottle.default_app()

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    STATIC_ROOT = os.path.join(PROJECT_ROOT, 'static').replace('\\', '/')
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    
    PORT = 5310

    @bottle.route('/static/<filepath:path>')
    def server_static(filepath):
        """Handler for static files, used with the development server.
        When running under a production server such as IIS or Apache,
        the server should be configured to serve the static files."""
        return bottle.static_file(filepath, root=STATIC_ROOT)

    # Starts a local test server.
    bottle.run(server='wsgiref', host=HOST, port=PORT)
