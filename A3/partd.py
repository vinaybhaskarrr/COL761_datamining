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


def main(dimensions,input_data,k,dim_2,dim_4,dim_10,dim_20):

    values=[]

    for alpha in dimensions:

        pca = PCA(n_components=alpha,random_state=42)
        red_data = pca.fit_transform(input_data)
        m = red_data.shape[0]
        mtree = MTree(red_data)
        kdtree = KDTree(red_data, metric='euclidean')
        jjscore=0

        params_cp=falconn.get_default_parameters(m,alpha,falconn.DistanceFunction.EuclideanSquared)
        lsh = falconn.LSHIndex(params_cp)
        dataset = red_data.astype(np.float32)
        lsh.setup(dataset)

        points = np.random.choice(m, 100)
        sacn_list =[]
        for i in points:
            sacn_list.append(red_data[i])

        kdtree_time = []
        noindex_time = []
        lsh_time = []
        mtree_time = []

        for qpoint in sacn_list:

            qpoint_dimension =np.array([qpoint])
            initialtime = time.time()
            distances, indices = kdtree.query(qpoint_dimension, k = k)
            difftime = time.time()-initialtime
            kdtree_time.append(difftime)

            z = qpoint.astype(np.float32)
            initialtime = time.time()
            query_object = lsh.construct_query_object()
            nearest_neighbors = query_object.find_k_nearest_neighbors(z, k+200)
            difftime = time.time()-initialtime
            lsh_time.append(difftime)
            nearest_neighbors=nearest_neighbors[:k]

            jjj = jaccard_score(np.array(indices).flatten(), np.array(nearest_neighbors).flatten(), average = 'weighted')
            jjscore = jjscore+jjj

        jjscore/=len(sacn_list)
        # values.append(jjscore)
        if(alpha==2):
            dim_2.append(jjscore)
        if(alpha==4):
            dim_4.append(jjscore)
        if(alpha==10):
            dim_10.append(jjscore)
        if(alpha==20):
            dim_20.append(jjscore)
 

    # plt.plot(dimensions,values)
    # plt.ylabel('Average jaccard_score', fontweight ='bold', fontsize = 15)
    # plt.xlabel('dimensions', fontweight ='bold', fontsize = 15)
    # plt.legend()
    # plt.savefig (str(k)+".png")
    # plt.clf()


if __name__ == "__main__":
    inputfile = sys.argv[1]
    input_data = pd.read_csv(inputfile,header=None,sep=" ")
    input_data = np.array(input_data)
    input_data = input_data[:,:-1]
    dimensions = [2,4,10,20]
    dim_2=[]
    dim_4=[]
    dim_10=[]
    dim_20=[]

    k_list = [1,5,10,50,100,500]

    for k in k_list:
        main(dimensions,input_data,k,dim_2,dim_4,dim_10,dim_20)
    
    plt.plot(k_list,dim_2)
    plt.ylabel('jaccard_score', fontweight ='bold', fontsize = 15)
    plt.xlabel('K-Values', fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.savefig ("dimension2.png")
    plt.clf()

    plt.plot(k_list,dim_4)
    plt.ylabel('jaccard_score', fontweight ='bold', fontsize = 15)
    plt.xlabel('K-Values', fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.savefig ("dimension4.png")
    plt.clf()

    plt.plot(k_list,dim_10)
    plt.ylabel('jaccard_score', fontweight ='bold', fontsize = 15)
    plt.xlabel('K-Values', fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.savefig ("dimension10.png")
    plt.clf()

    plt.plot(k_list,dim_20)
    plt.ylabel('jaccard_score', fontweight ='bold', fontsize = 15)
    plt.xlabel('K-Values', fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.savefig ("dimension20.png")
    plt.clf()
    
    plt.plot(k_list, dim_2, label='D_2')
    plt.plot(k_list, dim_4, label='D_4')
    plt.plot(k_list, dim_10, label='D_10')
    plt.plot(k_list, dim_20, label='D_20')
    plt.title('jacard_score on Variation of K ')
    plt.xlabel('K-values', fontweight='bold', fontsize=15)
    plt.ylabel('jacard_score(AVG)', fontweight='bold', fontsize=15)
    plt.legend()
    plt.savefig("All dimensions.png")
    plt.clf()






