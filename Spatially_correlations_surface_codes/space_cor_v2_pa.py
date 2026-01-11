import numpy as np
from scipy.linalg import block_diag
import pickle

import itertools
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
np.random.seed(rank)

def find_neighbors_det(coord_list, target_coord, max_offset=1):
    x0, y0, z0 = target_coord
    neighbors = [c for c in coord_list
                 if c[2] == z0 and
                    abs(c[0] - x0)+abs(c[1] - y0) <= max_offset and
                    c != target_coord]
    if z0<d-1:
        neighbors_sup = [c for c in coord_list
                        if c[0] == x0 and
                        c[1] == y0 and
                        z0 - c[2] >= 0 and
                        z0 -c[2] <=1
                        ]
    else:
        neighbors_sup = [c for c in coord_list
                        if c[0] == x0 and
                        c[1] == y0 and
                        z0 - c[2] > 0 and
                        z0 -c[2] <=1
                        ]
    return neighbors+neighbors_sup

def find_neighbors_box(coord_list, target_coord, max_offset=2):
    x0, y0, z0 = target_coord
    neighbors = [c for c in coord_list
                 if c[2] == z0 and
                    abs(c[0] - x0) <= max_offset and
                    abs(c[1] - y0) <= max_offset and
                    c != target_coord]
    return neighbors

d = 7
width = 2*d-1
length = width*2+1
height = d


shots = 50000
# 示例：
coords = [(x, y, z) for x in range(length) for y in range(width) for z in range(height)]
coords = sorted(coords, key=lambda c: (c[2], c[1], c[0]))
map_coords_to_idx = {}
x = 0

for c in coords:
    if c[0] == 2*d - 1:
        None
    elif (c[1]+c[0])%2==1 and c[2]==d-1:
        None    
    else:
        map_coords_to_idx.update({c: x})
        x = x+1

det_coords = []
Hx_coords = []
for c in coords:
    if c[1]%2 ==1 and c[0]%2==0:
        det_coords.append(c)
        Hx_coords.append(find_neighbors_det(coords,c))

Hx_matrix = []
for i in range(len(det_coords)):
    hx_one = [0 for i in range(len(map_coords_to_idx))]
    Hx_coord = Hx_coords[i]
    for j in range(len(Hx_coord)):
        try:
            hx_one[map_coords_to_idx[Hx_coord[j]]] = 1
        except KeyError:
            continue
    
    Hx_matrix.append(hx_one)
Hx_matrix = np.array(Hx_matrix)

# # 找出全为0的列（按列判断）
# zero_col_mask = np.all(Hx_matrix == 0, axis=0)

# # 提取下标
# zero_col_indices = np.where(zero_col_mask)[0]

# Hx_matrix_new = np.delete(Hx_matrix, zero_col_indices, axis=1)
# print(Hx_matrix)

error_coords = list(map_coords_to_idx.keys())
neighbordic = {}
for c in error_coords:
    neighborlist = find_neighbors_box(error_coords,c)
    neighbordic.update({c:neighborlist})
plist = np.logspace(np.log10(0.01), np.log10(0.05), num=10)
p1p2 = []
p12 = []

for p0 in plist:
    # p0 = 0.02
    p_c = 0.1
    error_all = []
    for i in range(shots):
        error = [False for i in range(len(error_coords))]
        for c in error_coords:
            flip = np.random.binomial(1,p0)
            if flip ==1:
                error[map_coords_to_idx[c]] = not error[map_coords_to_idx[c]]
                if np.random.binomial(1,p_c) ==1:
                    flag = np.random.choice(24)
                    if flag < len(neighbordic[c]):
                        point = neighbordic[c][flag]
                        error[map_coords_to_idx[point]] = not error[map_coords_to_idx[point]]
        error_all.append(error)
    error_all = np.array(error_all)
    # error_all_new = np.delete(error_all, zero_col_indices, axis=1)
    det_all = (error_all@Hx_matrix.T)%2
    import pymatching

    matching = pymatching.Matching.from_check_matrix(check_matrix=Hx_matrix,error_probabilities=[p0 for x in range(len(Hx_matrix[0]))])
    e_hat = matching.decode_batch(det_all)

    q1_up_b = []
    q2_up_b = []
    q1_down_b = []
    q2_down_b = []
    for c in map_coords_to_idx:
        if c[1]==0:
            if c[0]%2 ==0:
                if c[0]<=2*d-1:
                    q1_up_b.append(map_coords_to_idx[c])
                else:
                    q2_up_b.append(map_coords_to_idx[c])
        elif c[1]==2*d-1-1:
            if c[0]%2 ==0:
                if c[0]<=2*d-1:
                    q1_down_b.append(map_coords_to_idx[c])
                else:
                    q2_down_b.append(map_coords_to_idx[c])
    matrix = (e_hat+error_all)%2
    non_zero_rows = [i for i in range(len(matrix)) if any(matrix[i])]


    num1 = 0
    num2 = 0
    num_adj = 0
    for row in non_zero_rows:
        # print(error_ind[row])
        # print(obs_data[row])
        err = matrix[row]
        flag1 = 0
        flag2 = 0
        if sum(err[q1_up_b])%2==1 and sum(err[q1_down_b])%2==1:
            num1+=1
            flag1 = 1
            
        if sum(err[q2_up_b])%2==1 and sum(err[q2_down_b])%2==1:
            num2+=1
            flag2 =1
        if flag1*flag2:
            num_adj+=1
    p1p2.append(num1*num2/(d*shots**2))
    p12.append(num_adj/(shots*d))



gathered_results1 = comm.gather(p1p2, root=0)
gathered_results2 = comm.gather(p12, root=0)

if rank == 0:
    P1P2 = [sum(values)/(size) for values in zip(*gathered_results1)]
    P12 = [sum(values)/(size) for values in zip(*gathered_results2)]
    res = list(zip(P12,P1P2))
    print(res)
    fname = './Result/D3_v3_d='+str(d)
    with open(fname, 'wb') as fp:
        pickle.dump(res, fp)
