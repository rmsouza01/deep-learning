# The code here is a translation of the MATLAB code provided at
# https://www.mathworks.com/matlabcentral/fileexchange/41459-6-functions-for-generating-artificial-datasets

import numpy as np

def spirals(N = 1000, degrees = 570, start = 90, noise = 0.2):
    """
    Generate "two spirals" dataset with N instances. 
    Input:
    N -> Number of samples in the dataset
    degrees -> controls the length of the spirals
    start -> determines how far from the origin the spirals start, in degrees
    noise displaces the instances from the spiral. 0 is no noise, at 1 the
    spirals will start overlapping
    Output:
    data -> (Nx3) dataset: (xcoord,ycoord,label)
    """
    
    deg2grad = (2*np.pi)/360
    start = start*deg2grad
    
    #Number of samples in spirals 1 and 2
    N1 = int(np.floor(1.0*N/2))
    N2 = N - N1
    
    #Spiral 1
    n = start + np.sqrt(np.random.rand(N1,1))*degrees*deg2grad
    d1x =  -np.cos(n)*n + np.random.rand(N1,1)*noise                   
    d1y = np.sin(n)*n + np.random.rand(N1,1)*noise    
    
    spiral1 = np.concatenate((d1x,d1y,\
                             np.zeros((N1,1))),axis = 1)
    
    # Spiral 2
    n = start + np.sqrt(np.random.rand(N2,1))*degrees*deg2grad
    d2x = np.cos(n)*n+np.random.rand(N2,1)*noise
    d2y = -np.sin(n)*n + np.random.rand(N2,1)*noise
    spiral2 = np.concatenate((d2x,d2y,\
                             np.ones((N2,1))),axis = 1)
    
    #Concatenating the final data
    data = np.concatenate((spiral1,spiral2),axis = 0)
    return data


def concentric_clusters(N = 1000, r1 = 1, r2 = 5, w1 = 0.8, w2 = 1.0/3, arms = 64):
    """
    Generates two concentric circular clusters.
    Input:
    N -> Number of samples in the dataset
    r1 -> Radius cluster 1
    r2 -> Radius cluster 2
    w1 -> Cluster 1 weight
    w2 -> Cluster 2 weight
    arms -> Number of stripes (arms) in the circle
    Output:
    data -> (Nx3) dataset: (xcoord,ycoord,label)
    """
    
    #Number of samples in each cluster
    N1 = int(np.floor(1.0*N/2))
    N2 = N - N1
    
    phi1 = np.random.rand(N1,1) * 2 * np.pi;
    dist1 = r1 + 1.0*(np.random.randint(0,5,size =(N1,1)))/5*w1*r1
    d1x = dist1* np.cos(phi1) 
    d1y = dist1* np.sin(phi1) 
    cluster1 = np.concatenate((d1x,d1y,\
                             np.zeros((N1,1))),axis = 1)
    
    perarm = int(np.round(N2/arms))
    N2 = perarm*arms;
    radperarm = (2*np.pi)/arms;
    phi2 = 1.0*(np.arange(1,N2+1,dtype = int) - np.arange(1,N2+1,\
            dtype=int)%perarm)/perarm*(radperarm)
    dist2 = r2 * (1 - w2/2) + (r2*w2*np.arange(1,N2+1,\
            dtype=int)%perarm)/perarm
    d2x = dist2*np.cos(phi2)
    d2y = dist2*np.sin(phi2)
    cluster2 = np.concatenate((d2x[:,np.newaxis],d2y[:,np.newaxis],\
                             np.ones((N2,1))),axis = 1)  
    
    #Concatenating the final data
    data = np.concatenate((cluster1,cluster2),axis = 0)
    return data



def corners(N = 100,scale = 0,gapwidth = 2,cornerwidth = 2):
    """
    Generates for "corner" clusters .
    Input:
    N -> Number of samples in the dataset
    scale -> Corner scale
    gapwidth -> gap between corners
    cornerwidth -> corner width
    Output:
    data -> (Nx3) dataset: (xcoord,ycoord,label)
    """
    
    N = int(N/8)*8
    per_corner  = N//4
    
    xplusmin = np.concatenate((np.ones((per_corner,1)),\
                               -np.ones((per_corner,1)),\
                               np.ones((per_corner,1)),\
                               -np.ones((per_corner,1))),\
                               axis = 0)
                              
    yplusmin = np.concatenate((np.ones((per_corner,1)),\
                               -np.ones((2*per_corner,1)),\
                               np.ones((per_corner,1))),\
                               axis = 0)
    
    #Horizontal edge
    x = xplusmin[::2]*gapwidth+xplusmin[::2]*scale*(np.random.rand(2*per_corner).reshape(-1,1))
    
    y = yplusmin[::2]*gapwidth + cornerwidth*yplusmin[::2]*(np.random.rand(2*per_corner).reshape(-1,1))                          
    
    z = np.floor(np.arange(2*per_corner,dtype = int)/(per_corner*0.5))
    
    horizontal = np.concatenate((x,y,z[:,np.newaxis]),axis = 1)
                  
    # Vertical edge
    x2 = xplusmin[1::2]*gapwidth + cornerwidth*xplusmin[1::2]*\
                                 (np.random.rand(2*per_corner).reshape(-1,1))
    
    y2 = yplusmin[1::2]*gapwidth+yplusmin[1::2]*scale*\
                                 (np.random.rand(2*per_corner).reshape(-1,1))
    
    z2 = np.floor(np.arange(2*per_corner,dtype = int)/(per_corner*0.5))
    vertical = np.concatenate((x2,y2,z2[:,np.newaxis]),axis = 1)
    
    #Concatenating the final data
    data = np.concatenate((vertical,horizontal),axis = 0)
    return data



def crescent_full_moon(N = 1000,r1 = 5,r2 = 10,r3 = 15):
    """
    Generates 2 clusters. One shape as a full moon and the other as a crescent moon.
    Input:
    N -> Number of samples in the dataset
    r1 -> 
    r2 -> 
    r3 -> 
    Output:
    data -> (Nx3) dataset: (xcoord,ycoord,label)
    """

    N = int(N/4)*4
    N1 = int(N/4)
    N2 = N-N1   
     
    # Full moon
    phi1 = np.random.rand(N1,1)*2*np.pi
    R1 = np.sqrt(np.random.rand(N1, 1))
    moon = np.concatenate((np.cos(phi1)*R1*r1,np.sin(phi1)*R1*r1,\
                           np.zeros((N1,1))),axis = 1)
    # Crescent moon
    d = r3 - r2
    phi2 = np.pi + np.random.rand(N2,1)*np.pi
    R2 = np.sqrt(np.random.rand(N2,1))
    crescent = np.concatenate((np.cos(phi2)*(r2 + R2 * d),\
                               np.sin(phi2)*(r2 + R2 * d),\
                               np.ones((N2,1))),axis = 1)
    #concatenating the final data
    data = np.concatenate((moon,crescent),axis = 0)
    return data


    
