from queue import PriorityQueue

"""
.qsize()  :: Return the approximate size of the queue. 
.empty()  :: Return True if the queue is empty, False otherwise. 
.full()   :: Return True if the queue is full, False otherwise.
.put()    :: Put item into the queue.
.get()    :: Remove and return an item from the queue.

- queue.PriorityQueue(maxsize=0)
    Constructor for a priority queue. 
    maxsize is an integer that sets the upperbound 
    limit on the number of items that can be placed in the queue. 
    Insertion will block once this size has been reached, 
    until queue items are consumed. 
    If maxsize is less than or equal to zero, the queue size is infinite.
"""

import numpy as np 
import random 
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import pynndescent

def cal_dist(a, b, metric='euclidean'):
    """
    implement it.
    """
    dst = -1e6
    if metric == "euclidean":
        dst = distance.euclidean(a,b)
    if metric == "cosine":
        dst = distance.cosine(a,b)
    return dst 

def load_fvecs(file="", dtype=np.float32, verbose=False):
    """
    this func is designed to read `.fvecs` binary files. 
    `.fvecs` :: The vectors are stored in raw little endian. 
    Each vector takes 4+d*4 bytes for .fvecs and .ivecs formats, and 4+d bytes for .bvecs formats, 
    where d is the dimensionality of the vector, as shown below. 
    source link: https://www.gsitechnology.com/ANN-Benchmarks-Data-Scientists-Journey-to-Billion-Scale-Performance
    source link: http://corpus-texmex.irisa.fr/#matlab
    """
    data     = np.fromfile(file=file, dtype=dtype)
    data_dim = data.view(np.int32)[0]
    data     = data.reshape(-1, data_dim + 1)[:, 1:]
    #------------------------------------------------#
    if verbose:
        print('data_name:\t ' + file)
        print('data_dim:\t{}'.format(data_dim))
        print('data_shape:\t {}*{}'.format(data.shape[0], data.shape[1]))
    return data

# def load_ivecs(file="", dtype=np.int32, verbose=False):
#     data     = np.fromfile(file=file, dtype=dtype)
#     data_dim = data.view(np.int32)[0]
#     data     = data.reshape(-1, data_dim + 1)[:, 1:]
#     #------------------------------------------------#
#     if verbose:
#         print('data_name:\t ' + file)
#         print('data_dim:\t{}'.format(data_dim))
#         print('data_shape:\t {}*{}'.format(data.shape[0], data.shape[1]))
#     return data


def NNSearch(data, query_vec, KNN, RKNN, N_topK):
    N = data.shape[0] 
    flag = [0 for _ in range(N)]
    MAX_DIST = 1e9 
    D = [MAX_DIST for _ in range(N)]
    p = 9                             ## size of random seeds 



    for _ in range(p):
        idx = random.randint(0, N-1) 
        print(idx)
        dst = cal_dist(data[idx], query_vec)
        R.put(item=(dst,  idx))   ## .get :: get the idx with nearest dist 
        Q.put(item=(-dst, idx))

    # tmp_size = R.qsize()
    # for _ in range(tmp_size):
    #     _, idx = R.get() ## loop 
    #     print(idx) 
    #     flag[idx] = 1 
    #     D[idx] = cal_dist(data[idx], query_vec)
    #     #FIXME: at this moment, Q is empty, so the return value is NULL. 
    #     if Q.empty():
    #         Q.put(item=(-D[idx], idx))
    #         R.put(item=(D[idx],  idx))
    #     else: ## not empty 
    #         _, q_idx = Q.get() 
    #         max_q_dst = cal_dist(data[q_idx], query_vec)
    #         print(q_idx, max_q_dst, Q.qsize())
    #         if D[idx] < max_q_dst or Q.qsize() < N_topK:
    #             Q.put(item=(-max_q_dst, q_idx))
    #             R.put(item=(D[idx],  idx))

    tmp = [[] for _ in range(N)]
    for i, NN in enumerate( KNN):
        for nb in NN:
            tmp[i].append(nb)

    KNN = tmp 

    while (not R.empty()):
        min_r_dst, r_idx = R.get() ## nearest
        # print('near ', min_r_dst, r_idx)
        max_q_dst, q_idx = Q.get() ## largest 
        Q.put((max_q_dst, q_idx)) ## pop && put
        print('far  ', max_q_dst, q_idx)
        if min_r_dst > abs(max_q_dst):
        # if cal_dist(data[r_idx], query_vec) > cal_dist(data[q_idx], query_vec):
            break
        union_nb_hood = set(KNN[r_idx] + RKNN[r_idx])
        for nb in list(union_nb_hood):
            # print(nb)
            if flag[nb] == 0:
                flag[nb] = 1
                D[nb]    = cal_dist(data[nb], query_vec)
                # _, q_idx = Q.get() 

                max_q_dst, q_idx = Q.get() ## largest 
                Q.put((max_q_dst, q_idx)) ## pop && put

                if Q.qsize() < N_topK:
                    # dst = cal_dist(data[nb], query_vec)
                    Q.put(item=(-D[nb], nb))
                    # dst = cal_dist(data[q_idx], query_vec)
                    R.put(item=(D[nb],  nb))
                else:
                    if D[nb] < abs(max_q_dst):
                        _, _ = Q.get() ## pop largest
                        Q.put(item=(-D[nb], nb))
                        # dst = cal_dist(data[q_idx], query_vec)
                        R.put(item=(D[nb],  nb))


    res = []
    while not Q.empty():
        _, idx = Q.get() ## largest
        res.append(idx)
    
    return res 



if __name__ == "__main__":
    N_topK = 10 
    Q = PriorityQueue(maxsize=N_topK)  ## fixed-size PQ  :: return largest
    R = PriorityQueue(maxsize=0)       ## infinite-size  :: return smallest 



    data_folder = r"/Users/jeff/Downloads/sift1m/"
    sift1m_pth  = data_folder + 'sift_base.fvecs'
    sift1m      = load_fvecs(sift1m_pth)

    sift1m_query_pth = data_folder + 'sift_query.fvecs'
    sift1m_query     = load_fvecs(sift1m_query_pth)

    sift1m_gt_pth = data_folder + 'sift_groundtruth.ivecs'
    sift1m_gt     = load_fvecs(sift1m_gt_pth, dtype=np.int32, verbose=True) ## indexes should be `int`
    
    
    indices = pynndescent.NNDescent(sift1m, n_neighbors = N_topK)
    indices = indices.neighbor_graph[0]
    KNN = indices[:,1:]
    KNN = list(KNN)

    N = len(KNN)

    ## convert KNN into RKNN
    RKNN = [[] for _ in range(N)]
    for i, NN in enumerate(KNN):
        for nb in NN:
            RKNN[nb].append(i) 

    res = NNSearch(data=sift1m, query_vec=sift1m_query[0], KNN=KNN, RKNN=RKNN, N_topK=10)

    hit = 0
    for nb in res:
        if nb in sift1m_gt[0][:10]:
            hit += 1
    print('recall:', hit / 10)


    """
    One main problem is once the PQ reachs its maximum size, we canno put smoothly in few seconds.
    """