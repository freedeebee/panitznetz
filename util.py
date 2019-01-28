
# Dataset 1: quarter circle
import matplotlib.pyplot as plt
import pylab
import numpy as np
from PIL import Image


class Dataset(object):
    ''' abstract class for synthetic binary datasets '''

    def __init__(self, n=1000):
        '''n = number of samples to synthesize. '''
        self.n = n
    
    def get(self, batchsize):
        '''returns either a batch of samples, or the whole dataset.
           Each sample x is a 2D vector and comes with a 2D one-hot label.'''
        if batchsize is None:
            return self.X, self.t
        else:
            idx = np.array(random.sample(range(self.n), batchsize))
            X_ = self.X[idx]
            t_ = self.t[idx]
            return X_, t_


class DatasetCircle(Dataset):
    ''' a two-class dataset with a circle in the center. '''

    def __init__(self, n=1000):
        super(DatasetCircle, self).__init__(n)
        self.X = -1.5 + np.random.random((n,2)) * 3.0
        self.t = np.array([([1,0] if np.dot(x,x)<0.75 else [0,1]) for x in self.X])

        
class DatasetLogo(Dataset):
    ''' a two-class dataset with the HSRM logo in the center. '''

    def read_jpg(self, path):
        return np.asarray(Image.open(path))

    def __init__(self, n=1000):
        super(DatasetLogo, self).__init__(n)

        img = self.read_jpg("logo.jpg")[:,:,0].T
        n0,n1 = img.shape

        self.X = np.random.random((n,2)) * 2 - 1
        self.t = np.array([([1,0] if img[int((x0-(-1))/2.*n0),
                                         int((x1-(-1))/2.*n1)]>128 else [0,1])
                           for [x0,x1] in self.X])



def plot(model, X, t, path, xlim=[-1.5,1.5], ylim=[-1.5,1.5]):

    # define a figure to plot into
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # split samples into the two classes
    samples1 = [x for x,_t in zip(X,t) if _t[0]]
    samples0 = [x for x,_t in zip(X,t) if _t[1]]

    # define grid to plot
    x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/100.)
    y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/100.)

    # plot the model's score as a color map
    if model is not None:
        F = np.zeros((len(x), len(y)))
        
        for i,xval in enumerate(x):
            for j,yval in enumerate(y):
                y_ = model.predict(np.array([[xval,yval]]))
                F[i,j] = y_[0,1]

        im = ax.imshow(F.T, interpolation='bicubic', origin='lower',
                        cmap=pylab.get_cmap("RdYlGn"), extent=[xlim[0],xlim[1],ylim[0],ylim[1]]  )

    # plot the samples as dots
    if samples1 is not None:

        x1,y1 = list(zip(*samples1))
        x0,y0 = list(zip(*samples0))
        
        plt.plot(x1,y1, 'o', color='red', markersize=2)
        plt.plot(x0,y0, 'o', color='green', markersize=2)

    # configure and store the plot
    plt.xlim( xlim )
    plt.ylim( ylim )
    plt.grid()
    plt.tight_layout()
    plt.savefig(path)

