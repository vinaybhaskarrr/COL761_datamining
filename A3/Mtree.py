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

    mtree_mean = []
    mtree_std = []
    values=[]

    for alpha in dimensions:
        pca = PCA(n_components=alpha,random_state=42)
        red_data = pca.fit_transform(input_data)
        m = red_data.shape[0]
        mtree = MTree(red_data)
        points = np.random.choice(m, 100)
        sacn_list =[]
        for i in points:
            sacn_list.append(red_data[i])
        mtree_time = []

        for qpoint in sacn_list:
            qpoint_dimension =np.array([qpoint])
            

            initialtime = time.time()        
            nearest_neighbors = mtree.find_k_nearest_neighbors(qpoint, k)
            difftime=time.time()-initialtime
            mtree_time.append(difftime)


        mtree_mean.append(np.mean(mtree_time))
        mtree_std.append(np.std(mtree_time))


    
    print("mtree Mean",mtree_mean)
    print("mtree std",mtree_std)

    plt.errorbar(dimensions, mtree_mean, mtree_std, label='MTREE')
    plt.ylabel('Average Query Running time (sec)', fontweight ='bold', fontsize = 15)
    plt.xlabel('dimensions', fontweight ='bold', fontsize = 15)

    plt.legend()
    plt.savefig("Mtree.png")
    plt.clf()



if __name__ == "__main__":
    inputfile = sys.argv[1]
    input_data = pd.read_csv(inputfile,header=None,sep=" ")
    input_data = np.array(input_data)
    input_data = input_data[:,:-1]
    dimensions = [2,4,10,20]
    k= 5
    main(dimensions,input_data,k)


