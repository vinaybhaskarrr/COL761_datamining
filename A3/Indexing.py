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




class MTree:
    def __init__(self, input_data):
        self.tree = BallTree(input_data, metric='euclidean')

    def find_k_nearest_neighbors(self, query_point, k):
        distances, indices = self.tree.query([query_point], k=k)
        return indices[0]



def main(dimensions,input_data,k):

    kdtree_mean = []
    kdtree_std=[]
    mtree_mean = []
    mtree_std = []
    lsh_mean=[]
    lsh_std=[]
   

    for alpha in dimensions:

        pca = PCA(n_components=alpha,random_state=42)
        red_data = pca.fit_transform(input_data)
        m = red_data.shape[0]
        mtree = MTree(red_data)
        kdtree = KDTree(red_data, metric='euclidean')
        params_cp=falconn.get_default_parameters(m,alpha,falconn.DistanceFunction.EuclideanSquared)
        lsh = falconn.LSHIndex(params_cp)
        dataset = red_data.astype(np.float32)
        lsh.setup(dataset)

        points = np.random.choice(m, 100)
        sacn_list =[]
        for i in points:
            sacn_list.append(red_data[i])

        kdtree_time = []
        lsh_time = []
        mtree_time = []

        for qpoint in sacn_list:

            qpoint_dimension =np.array([qpoint])
            initialtime = time.time()
            help = kdtree.query(qpoint_dimension, k = k)
            distances = help[0]
            indices = help[1]
            difftime = time.time()-initialtime
            kdtree_time.append(difftime)

            initialtime = time.time()        
            nearest_neighbors = mtree.find_k_nearest_neighbors(qpoint, k)
            difftime=time.time()-initialtime
            mtree_time.append(difftime)

            z = qpoint.astype(np.float32)
            initialtime = time.time()
            query_object = lsh.construct_query_object()
            nearest_neighbors = query_object.find_k_nearest_neighbors(z, k)
            difftime = time.time()-initialtime
            lsh_time.append(difftime)

            

        kdtree_mean.append(np.mean(kdtree_time))
        mtree_mean.append(np.mean(mtree_time))
        lsh_mean.append(np.mean(lsh_time))
        lsh_std.append(np.std(lsh_time))
        mtree_std.append(np.std(mtree_time))
        kdtree_std.append(np.std(kdtree_time))


    print("kdtree Mean",kdtree_mean)
    print("kdtree std",kdtree_std)
    print("mtree Mean",mtree_mean)
    print("mtree std",mtree_std)
    print("LSH Mean",lsh_mean)
    print("LSH Std",lsh_std)

    plt.errorbar(dimensions, kdtree_mean, kdtree_std, label='KDTREE')
    plt.errorbar(dimensions, lsh_mean, lsh_std, label='LSH')
    plt.errorbar(dimensions, mtree_mean, mtree_std, label='MTREE')
    plt.ylabel('Average Query Running time (sec)', fontweight ='bold', fontsize = 15)
    plt.xlabel('dimensions', fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.savefig("Combined.png")
    plt.clf()



if __name__ == "__main__":
    inputfile = sys.argv[1]
    input_data = pd.read_csv(inputfile,header=None,sep=" ")
    input_data = np.array(input_data)
    input_data = input_data[:,:-1]
    dimensions = [2,4,10,20]
    k= 5
    main(dimensions,input_data,k)


