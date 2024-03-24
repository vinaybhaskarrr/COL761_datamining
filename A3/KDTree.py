import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.metrics import jaccard_score
import numpy as np
import sys
import time
import math
import falconn




def main(dimensions,input_data,k):

    kdtree_mean = []
    kdtree_std=[]
    
    for alpha in dimensions:

        pca = PCA(n_components=alpha,random_state=42)
        red_data = pca.fit_transform(input_data)
        m = red_data.shape[0]
        kdtree = KDTree(red_data, metric='euclidean')
        points = np.random.choice(m, 100)
        sacn_list =[]
        for i in points:
            sacn_list.append(red_data[i])

        kdtree_time = []
      
        for qpoint in sacn_list:

            qpoint_dimension =np.array([qpoint])
            initialtime = time.time()
            help = kdtree.query(qpoint_dimension, k = k)
            distances = help[0]
            indices = help[1]
            difftime = time.time()-initialtime
            kdtree_time.append(difftime)

            
        kdtree_mean.append(np.mean(kdtree_time))
        kdtree_std.append(np.std(kdtree_time))


    print("kdtree Mean",kdtree_mean)
    print("kdtree std",kdtree_std)

    plt.errorbar(dimensions, kdtree_mean, kdtree_std, label='KDTREE')
    plt.ylabel('Average Query Running time (sec)', fontweight ='bold', fontsize = 15)
    plt.xlabel('dimensions', fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.savefig("KDtree.png")
    plt.clf()


if __name__ == "__main__":
    inputfile = sys.argv[1]
    input_data = pd.read_csv(inputfile,header=None,sep=" ")
    input_data = np.array(input_data)
    input_data = input_data[:,:-1]
    dimensions = [2,4,10,20]
    k= 5
    main(dimensions,input_data,k)


