import pandas
import numpy as np
def softmax(x):
	return np.exp(x) / np.exp(x).sum()

def sigmoid(X):
   return 1/(1+np.exp(-X))
import bottle
import os
import sys
import pandas as pd

# routes contains the HTTP handlers for our server and must be imported.
app=bottle
if '--debug' in sys.argv[1:] or 'SERVER_DEBUG' in os.environ:
    # Debug mode will enable more verbose output in the console window.
    # It must be set at the beginning of the script.
    bottle.debug(True)

import os
from nilearn import datasets
atlas = datasets.fetch_atlas_msdl()
def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.

    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.

    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]

    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)

    return subs
def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 2D or 3D data.

    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.

    See also unpooling().
    '''
    m, n = mat.shape[:2]
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny-1)*stride+f, (nx-1)*stride+f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m-f)//stride*stride+f, :(n-f)//stride*stride+f, ...]

    view = asStride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(2, 3), keepdims=return_max_pos)

    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result)
        return result, pos
    else:
        return result
atlas_filename = "64/64/2mm/maps.nii.gz"

def n11(folder):
        import nibabel
        d19=[]
        for i in os.listdir(folder)[0:30]:
                img = nibabel.load(os.path.join(folder, i))
                a = np.array(img.dataobj,dtype=np.float32)
                d19.append(a)

        for i in  d19:
            print(i.shape)
            #i = np.asarray(i, dtype=np.float32, shape=((64, 64, 33, 112)))
            x2q=        poolingOverlap(np.array(i,dtype=np.float32),12,3)
            x3q=np.array(poolingOverlap(x2q,2,2))
            x4q=np.array(poolingOverlap(x3q,2,2))
            jp=np.array(poolingOverlap(x4q,3,3))

            #qare=MyCustomSGD(jp,1e-6,20,2,1,0)


           # MyCustomSGD(np.array(poolingOverlap(x4q,3,3)[0][0]),0.001,50,2,5)
            #poolingOverlap(i,12,3)
            M1.append(sigmoid(np.array(jp,dtype=np.float64)))
            #m2.append(pandas.DataFrame(sigmoid(np.array(net2(xa),dtype=np.float64))[0][0]))
        return pandas.DataFrame(M1).to_html()

@bottle.route("/c")
def controlgru():
   return n11("control")

@bottle.route("/random/<size>")
def r(size):
    counter=[]
    for ih in range(100):
        sa=[]
        s=size.split("_")
        for i in s:
            sa.append(int(i))
        i=np.random.normal(size=tuple(sa))
        poolingOverlap(i,2,2)
        x3q=(np.array((poolingOverlap(i,8,8)))-np.array((poolingOverlap(i,8,8)).min()))/(np.array((poolingOverlap(i,8,8)).max()-np.array((poolingOverlap(i,8,8)).min())))
        x4q=np.array(poolingOverlap(x3q,5,2,method="mean"))
        x4q=(x4q- x4q.min())/ (x4q.max() - x4q.min())
#        jp=np.array(poolingOverlap((x4q- x4q.min())/ (
# .max() - x4q.min()),10,10)[0][0])[0][1:len(poolingOverlap((x4q- x4q.min())/ (x4q.max() - x4q.min()),10,10)[0][0])-2]
       # JP2=np.array(poolingOverlap((x4q- x4q.min())/ (x4q.max() - x4q.min()),3,8))[0][0]        
        x4q=poolingOverlap(x4q[0][0],4,4)       # qare=MyCustomSGD(jp,1e-3,20,2,1,1)
#        np.dot(np.swapaxes(qare[0],0,1),jp[1:jp.shape[0]-2])
        xt=poolingOverlap(poolingOverlap(x4q,1,6,"mean"),1,2)
        xt.swapaxes(0,1)
        ju=[]
        for i in xt:
            ju.append(np.max(i))
        xt=np.resize(np.array(ju),(2,2))  
        
        #      xqi8=(xt-xt.min())/(xt.max()-xt.min())

        #xqi8=(xt-xt.min())/(xt.max()-xt.min())
        counter.append(xt[0])
    x=np.array(counter)
    counter.append(dict(avg=x.mean(),std=x.std()))
    return pd.DataFrame(counter).to_html()
@bottle.route("/net/test/<path>")
def tester(path):
        import nibabel,dicom2nifti
        dicom2nifti.dicom_series_to_nifti(path,path+"/_o1.nii.gz")
        img = nibabel.load(path+"/_o1.nii.gz")
        i = np.array(img.dataobj)
        #i.reshape((64,64,33,112))
        i=poolingOverlap(i,2,2)
        x3q=(np.array((poolingOverlap(i,8,8)))-np.array((poolingOverlap(i,8,8)).min()))/(np.array((poolingOverlap(i,8,8)).max()-np.array((poolingOverlap(i,8,8)).min())))
        x4q=np.array(poolingOverlap(x3q,5,2,method="mean"))
        x4q=(x4q- x4q.min())/ (x4q.max() - x4q.min())
#        jp=np.array(poolingOverlap((x4q- x4q.min())/ (
# .max() - x4q.min()),10,10)[0][0])[0][1:len(poolingOverlap((x4q- x4q.min())/ (x4q.max() - x4q.min()),10,10)[0][0])-2]
       # JP2=np.array(poolingOverlap((x4q- x4q.min())/ (x4q.max() - x4q.min()),3,8))[0][0]        
        x4q=poolingOverlap(x4q[0][0],4,4)       # qare=MyCustomSGD(jp,1e-3,20,2,1,1)
#        np.dot(np.swapaxes(qare[0],0,1),jp[1:jp.shape[0]-2])
        xt=poolingOverlap(poolingOverlap(x4q,1,6,"mean"),1,2)
        xt.swapaxes(0,1)
        ju=[]
        for i in xt:
            ju.append(np.max(i))
        xt=np.resize(np.array(ju),(2,2))  
        
        #      xqi8=(xt-xt.min())/(xt.max()-xt.min())

        #xqi8=(xt-xt.min())/(xt.max()-xt.min())
        return str(img.dataobj.shape)+"<hr>"+str(xt[0])
        (dict(avg=xqi8.mean(),std=xqi8.std(),softmax=softmax(xqi8),SOFTMAXAVG=softmax(xqi8).mean()))

@bottle.route("/s")
def schizogro():
   return n11("out/")

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
