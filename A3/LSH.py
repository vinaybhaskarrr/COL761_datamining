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
    lsh_mean=[]
    lsh_std=[]

    for alpha in dimensions:
        pca = PCA(n_components=alpha,random_state=42)
        red_data = pca.fit_transform(input_data)
        m = red_data.shape[0]
        params_cp=falconn.get_default_parameters(m,alpha,falconn.DistanceFunction.EuclideanSquared)
        lsh = falconn.LSHIndex(params_cp)
        dataset = red_data.astype(np.float32)
        lsh.setup(dataset)

        points = np.random.choice(m, 100)
        sacn_list =[]
        for i in points:
            sacn_list.append(red_data[i])

        lsh_time = []
        for qpoint in sacn_list:

            qpoint_dimension =np.array([qpoint])
            z = qpoint.astype(np.float32)
            initialtime = time.time()
            query_object = lsh.construct_query_object()
            nearest_neighbors = query_object.find_k_nearest_neighbors(z, k)
            difftime = time.time()-initialtime
            lsh_time.append(difftime)

            
        lsh_mean.append(np.mean(lsh_time))
        lsh_std.append(np.std(lsh_time))
        


    print("LSH Mean",lsh_mean)
    print("LSH Std",lsh_std)

    
    plt.errorbar(dimensions, lsh_mean, lsh_std, label='LSH')
    plt.ylabel('Average Query Running time (sec)', fontweight ='bold', fontsize = 15)
    plt.xlabel('dimensions', fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.savefig("LSH.png")
    plt.clf()



if __name__ == "__main__":
    inputfile = sys.argv[1]
    input_data = pd.read_csv(inputfile,header=None,sep=" ")
    input_data = np.array(input_data)
    input_data = input_data[:,:-1]
    dimensions = [2,4,10,20]
    k= 5
    main(dimensions,input_data,k)


