import numpy as np
from  tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
import heapq
import torch
import static.variable as VAR
import itertools
from scipy.spatial import cKDTree

device = torch.device(VAR.CUDA)

def coarse_est_spf(pts,lines):
    num_pts = pts.shape[0]    
    num_nn_l2l = int(min(500,0.05*num_pts))
    
    print("Calculating l2l neighbours")
    _nn_l2l = np.zeros([num_pts,num_nn_l2l],dtype=np.int32)
    nn_l2l = calc_l2l_nn(pts,lines,_nn_l2l,num_pts,num_nn_l2l)
    
    print("Calculating peak.")
    est_peak = estimate_all_pts_one_peak(pts,lines, nn_l2l)
    errs = np.abs(est_peak)
    pts_est = pts + est_peak.reshape(-1, 1) * lines

    print("Coarse estimation done.")
    print("Mean error : {}".format(np.mean(errs)))
    print("Median error : {}".format(np.median(errs)))
    # pts_est = pts + est_peak.reshape(-1, 1) * lines
    return est_peak, pts_est

def coarse_est_spf_harsh(points3d,lines,ind_to_ids,swap_levels):
    pts, pts2 = points3d
    ind_to_id1, ind_to_id2 = ind_to_ids
    swapped_ind_id1, swapped_id_ind1 = [], []
    swapped_ind_id2, swapped_id_ind2 = [], []
    num_pts = pts.shape[0]    
    num_nn_l2l = int(min(1000,0.05*num_pts))
    
    print("Calculating l2l neighbours")
    _nn_l2l = np.zeros([num_pts,num_nn_l2l],dtype=np.int32)
    nn_l2l = calc_l2l_nn(pts,lines,_nn_l2l,num_pts,num_nn_l2l)
    
    print("Calculating peak.")
    # Beta Peak Location
    dist = np.linalg.norm(np.subtract(pts2, pts), axis=1)
    est_peak = estimate_all_pts_one_peak(pts,lines, nn_l2l)
    inv_est_peak = dist - est_peak
    close_choice = est_peak<=inv_est_peak
    closeid = np.where(close_choice,list(ind_to_id1.values()),list(ind_to_id2.values()))
    distantid = np.where(close_choice,list(ind_to_id2.values()),list(ind_to_id1.values()))
    
    print("Coarse estimation done.")
    
    print("Start swap")
    for swap in swap_levels:
        new_ind_id1, new_ind_id2 = {},{}
        new_id_ind1, new_id_ind2 = {},{}
        swapped_setA,swapped_setB = swap_harsh_spf(closeid,distantid,swap)
        for i in range(num_pts):
            new_ind_id1[i] = swapped_setA[i]
            new_id_ind1[swapped_setA[i]] = i
            new_ind_id2[i] = swapped_setB[i]
            new_id_ind2[swapped_setB[i]] = i
        swapped_ind_id1.append(new_ind_id1)
        swapped_id_ind1.append(new_id_ind1)
        swapped_ind_id2.append(new_ind_id2)
        swapped_id_ind2.append(new_id_ind2)
    
    # TODO print errors w.r.t swapped points
    # errs_a = np.linalg.norm(np.subtract(pts_est1,pts),axis=1)
    # errs_b = np.linalg.norm(np.subtract(pts_est2,pts2),axis=1)
    
    # print("Errors")
    # print("1st Point Mean error : {}".format(np.mean(errs_a)))
    # print("1st Point Median error : {}".format(np.median(errs_a)))
    # print("2nd Point Mean error : {}".format(np.mean(errs_b)))
    # print("2nd Point Median error : {}".format(np.median(errs_b)))
    # print()
    
    return est_peak, [swapped_ind_id1,swapped_ind_id2], [swapped_id_ind1,swapped_id_ind2]
    
def swap_harsh_spf(close,distant,swap):
    setA,setB = np.array([]),np.array([])
    if swap==1:
        print("100% swap")
        setA, setB = distant, close
    elif swap==0:
        print("No swap")
        setA, setB = close, distant
    else:
        print(f"{100*swap}% swap")
        n = len(close)
        randchoice = np.ones(n)
        rand_ind= np.random.permutation(n)[:int(n*swap)]
        randchoice[rand_ind]=0
        setA = np.where(randchoice==1,close,distant)
        setB = np.where(randchoice==0,close,distant)

    return setA, setB

def calc_max_dist(pts,pts2,num_pts):
    max_d = -1e9
    pts_temp = np.concatenate((pts, pts2))
    temp = torch.zeros(num_pts*2).to(device)
    pts_temp = torch.from_numpy(pts_temp).to(device)
    for i in tqdm(range(num_pts*2)):
        temp[i] = torch.max(torch.norm(pts_temp - torch.roll(pts_temp, i, 0), dim=1))
    max_d = torch.max(temp).cpu().numpy()
    X_MAX = max_d * 0.75
    return X_MAX

def calc_l2l_nn(pts,lines,nn_l2l,num_pts,num_nn_l2l):
    pts = torch.from_numpy(pts).to(device)
    lines = torch.from_numpy(lines).to(device)
    nn_l2l = torch.from_numpy(nn_l2l).to(device)
    for i in tqdm(range(num_pts)):
        nn_l2l[i] = get_n_closest_lines_from_line_torch(pts[i].repeat(num_pts,1),lines[i].repeat(num_pts,1),pts,lines,num_nn_l2l)
    pts = pts.cpu().numpy()
    lines = lines.cpu().numpy()
    nn_l2l = nn_l2l.cpu().numpy()
    return nn_l2l

def coarse_est_tpf(points3d, lines, swap_level):
    pts, pts2 = points3d
    num_pts = pts.shape[0]
    num_nn_l2l = int(min(1000,0.05*num_pts))
    
    print("Calculating max distance")
    maxDist = calc_max_dist(pts,pts2,num_pts)
    print("Calculating l2l neighbours")
    _nn_l2l = np.zeros([num_pts,num_nn_l2l],dtype=np.int32)
    nn_l2l = calc_l2l_nn(pts,lines,_nn_l2l,num_pts,num_nn_l2l)
    
    print("Calculating peak.")
    # Beta Peak Location
    gt_beta_B = np.linalg.norm(np.subtract(pts2, pts), axis=1)
    est_peak1, est_peak2 = estimate_all_pts_two_peaks(pts,lines, nn_l2l, gt_beta_B, maxDist)
    print("Coarse estimation done.")
    
    est1 = pts + est_peak1.reshape(-1, 1) * lines
    est2 = pts + est_peak2.reshape(-1, 1) * lines
    
    print("Start swap")
    ests_pts=[]
    for swap in swap_level:
        if swap==1:
            print("100% swap")
            pts_est1, est_peak1, pts_est2, est_peak2 = est2, est_peak2, est1, est_peak1
        elif swap==0:
            print("No swap")
            pts_est1, pts_est2 = est1, est2
        else:
            print(f"{100*swap}% swap")
            randchoice = np.ones(len(pts))
            rand_ind= np.random.permutation(len(pts))[:int(len(pts)*swap)]
            randchoice[rand_ind]=0
            random_choice = np.repeat(randchoice,3).reshape(-1,3)
            pts_est1 = np.where(random_choice==1,est1,est2) # in order
            pts_est2 = np.where(random_choice==0,est1,est2) # swapped
            
        ests_pts.append([pts_est1, pts_est2])
        errs_a = np.linalg.norm(np.subtract(pts_est1,pts),axis=1)
        errs_b = np.linalg.norm(np.subtract(pts_est2,pts2),axis=1)
        
        print("Errors")
        print("1st Point Mean error : {}".format(np.mean(errs_a)))
        print("1st Point Median error : {}".format(np.median(errs_a)))
        print("2nd Point Mean error : {}".format(np.mean(errs_b)))
        print("2nd Point Median error : {}".format(np.median(errs_b)))
        print()
    
    return ests_pts

def refine_est_spf(pts,lines,est_peak,iter_num):
    num_pts = pts.shape[0]
    num_nn_l2p = 100
    num_nn_p2l = 100
    
    print("Refine estimation starts.")
    
    for i in range(iter_num):
        pts_est = pts + est_peak.reshape(-1, 1) * lines
        
        nn_l2p = np.zeros([num_pts, num_nn_l2p], dtype=np.int32)
        nn_p2l = np.zeros([num_pts, num_nn_p2l], dtype=np.int32)

        pts = torch.from_numpy(pts).to(device)
        pts_est = torch.from_numpy(pts_est).to(device)
        lines = torch.from_numpy(lines).to(device)
        nn_l2p = torch.from_numpy(nn_l2p).to(device)
        nn_p2l = torch.from_numpy(nn_p2l).to(device)

        print("Calculating l2p neighbours")
        for i in range(num_pts):
            nn_l2p[i, :] = get_n_closest_points_from_line_torch(pts[i, :].repeat(num_pts,1), lines[i, :].repeat(num_pts,1), pts_est, num_nn_l2p)

        print("Calculating p2l neighbours")
        for i in range(num_pts):
            nn_p2l[i, :] = get_n_closest_lines_from_point_torch(pts_est[i, :].repeat(num_pts,1), pts, lines, num_nn_p2l)
        pts = pts.cpu().numpy()
        pts_est = pts_est.cpu().numpy()
        lines = lines.cpu().numpy()
        nn_l2p = nn_l2p.cpu().numpy()
        nn_p2l = nn_p2l.cpu().numpy()

        nns = {}

        print("Finding refined estimates using intersection when possible")
        print(num_pts)
        for i in range(num_pts):

            set_p2l = set(nn_p2l[i, :])
            set_l2p = set(nn_l2p[i, :])
            set_intersection = set_p2l.intersection(set_l2p)

            if len(set_intersection) > 10:  # Threshld can be changed as well. A distance metric combining both l2p and p2l can also be defined.
                nns[i] = np.array(list(set_intersection), dtype=np.int32)
            else:
                nns[i] = nn_l2p[i, :]
                
        # TODO error w.r.t swapped points
        est_peak = estimate_all_pts_one_peak(pts, lines, nns)
        errs = np.abs(est_peak)

        print("Mean error : {}".format(np.mean(errs)))
        print("Median error : {}".format(np.median(errs)))
        print(f"Refine iteration {i} finished")

    pts_est = pts + est_peak.reshape(-1, 1) * lines
    
    return pts_est

def refine_est_spf_harsh(points3d,lines,ind_to_ids,swap_levels,est_peak,iter_num):
    pts, pts2 = points3d
    ind_to_id1, ind_to_id2 = ind_to_ids
    swapped_ind_id1, swapped_id_ind1 = [], []
    swapped_ind_id2, swapped_id_ind2 = [], []
    num_pts = pts.shape[0]    
    num_nn_l2l = int(min(1000,0.05*num_pts))
    num_nn_l2p = 100
    num_nn_p2l = 100
    
    print("Refine estimation starts.")
    
    for i in range(iter_num):
        pts_est = pts + est_peak.reshape(-1, 1) * lines
        
        nn_l2p = np.zeros([num_pts, num_nn_l2p], dtype=np.int32)
        nn_p2l = np.zeros([num_pts, num_nn_p2l], dtype=np.int32)

        pts = torch.from_numpy(pts).to(device)
        pts_est = torch.from_numpy(pts_est).to(device)
        lines = torch.from_numpy(lines).to(device)
        nn_l2p = torch.from_numpy(nn_l2p).to(device)
        nn_p2l = torch.from_numpy(nn_p2l).to(device)

        print("Calculating l2p neighbours")
        for i in range(num_pts):
            nn_l2p[i, :] = get_n_closest_points_from_line_torch(pts[i, :].repeat(num_pts,1), lines[i, :].repeat(num_pts,1), pts_est, num_nn_l2p)

        print("Calculating p2l neighbours")
        for i in range(num_pts):
            nn_p2l[i, :] = get_n_closest_lines_from_point_torch(pts_est[i, :].repeat(num_pts,1), pts, lines, num_nn_p2l)
        pts = pts.cpu().numpy()
        pts_est = pts_est.cpu().numpy()
        lines = lines.cpu().numpy()
        nn_l2p = nn_l2p.cpu().numpy()
        nn_p2l = nn_p2l.cpu().numpy()

        nns = {}

        print("Finding refined estimates using intersection when possible")
        print(num_pts)
        for i in range(num_pts):

            set_p2l = set(nn_p2l[i, :])
            set_l2p = set(nn_l2p[i, :])
            set_intersection = set_p2l.intersection(set_l2p)

            if len(set_intersection) > 10:  # Threshlod can be changed as well. A distance metric combining both l2p and p2l can also be defined.
                nns[i] = np.array(list(set_intersection), dtype=np.int32)
            else:
                nns[i] = nn_l2p[i, :]
                
        # TODO error w.r.t swapped points
        est_peak = estimate_all_pts_one_peak(pts, lines, nns)
        errs = np.abs(est_peak)

        print("Mean error : {}".format(np.mean(errs)))
        print("Median error : {}".format(np.median(errs)))
        print(f"Refine iteration {i} finished")
    
    dist = np.linalg.norm(np.subtract(pts2, pts), axis=1)
    inv_est_peak = dist - est_peak
    close_choice = est_peak<=inv_est_peak
    closeid = np.where(close_choice,list(ind_to_id1.values()),list(ind_to_id2.values()))
    distantid = np.where(close_choice,list(ind_to_id2.values()),list(ind_to_id1.values()))
    
    print("Coarse estimation done.")
    
    print("Start swap")
    for swap in swap_levels:
        new_ind_id1, new_ind_id2 = {},{}
        new_id_ind1, new_id_ind2 = {},{}
        swapped_setA,swapped_setB = swap_harsh_spf(closeid,distantid,swap)
        for i in range(num_pts):
            new_ind_id1[i] = swapped_setA[i]
            new_id_ind1[swapped_setA[i]] = i
            new_ind_id2[i] = swapped_setB[i]
            new_id_ind2[swapped_setB[i]] = i
        swapped_ind_id1.append(new_ind_id1)
        swapped_id_ind1.append(new_id_ind1)
        swapped_ind_id2.append(new_ind_id2)
        swapped_id_ind2.append(new_id_ind2)
    
    # TODO print errors w.r.t swapped points
    # errs_a = np.linalg.norm(np.subtract(pts_est1,pts),axis=1)
    # errs_b = np.linalg.norm(np.subtract(pts_est2,pts2),axis=1)
    
    # print("Errors")
    # print("1st Point Mean error : {}".format(np.mean(errs_a)))
    # print("1st Point Median error : {}".format(np.median(errs_a)))
    # print("2nd Point Mean error : {}".format(np.mean(errs_b)))
    # print("2nd Point Median error : {}".format(np.median(errs_b)))
    # print()

    pts_est = pts + est_peak.reshape(-1, 1) * lines
    
    return pts_est, [swapped_ind_id1,swapped_ind_id2], [swapped_id_ind1,swapped_id_ind2]

def point_distance(x, y):
    return np.linalg.norm(np.subtract(y, x))

def get_n_closest_lines_from_line_torch(pt, line, pts, lines, num_nn):
    num_pts = len(pts)
    
    n = torch.cross(line, lines)
    n /= torch.linalg.norm(n, axis=1, keepdims=True) + 10e-7

    dist = torch.abs(torch.sum(torch.multiply(pts - pt, n), axis=1)) 
    _,ii_nn = torch.topk(dist, num_nn + 1, largest=False)

    return ii_nn[1:num_nn+1]

def get_n_closest_rays_from_ray_torch(pt, line, pts, lines, num_nn):
    num_pts = len(pts)
    
    n = torch.cross(line, lines)
    n /= torch.linalg.norm(n, axis=1, keepdims=True) + 10e-7

    dist = torch.abs(torch.sum(torch.multiply(pts - pt, n), axis=1)) 
    _,ii_nn = torch.topk(dist, num_nn, largest=False)

    return ii_nn[:num_nn]

def get_n_closest_points_from_line_torch(pt, line, pts, num_nn):
    num_pts = len(pts)
    
    n = torch.cross(pts - pt, line)
    n /= torch.linalg.norm(n, axis=1, keepdims=True) + 10e-7
    
    n1 = torch.cross(n, line)
    n1 /= torch.linalg.norm(n1, axis=1, keepdims=True) + 10e-7

    dist = torch.abs(torch.sum(torch.multiply(pts - pt, n1), axis=1)) 
    _,ii_nn = torch.topk(dist, num_nn + 1, largest=False)

    return ii_nn[1:num_nn + 1]

def get_n_closest_points_from_ray_torch(pt, line, pts, num_nn):
    num_pts = len(pts)
    
    n = torch.cross(pts - pt, line)
    n /= torch.linalg.norm(n, axis=1, keepdims=True) + 10e-7
    
    n1 = torch.cross(n, line)
    n1 /= torch.linalg.norm(n1, axis=1, keepdims=True) + 10e-7

    dist = torch.abs(torch.sum(torch.multiply(pts - pt, n1), axis=1)) 
    _,ii_nn = torch.topk(dist, num_nn, largest=False)

    return ii_nn[:num_nn]

def get_n_closest_lines_from_point_torch(pt, pts, lines, num_nn):
    num_pts = len(pts)
    
    n = torch.cross(pts-pt,lines)
    n /= torch.linalg.norm(n, axis=1, keepdims=True) + 10e-7
    
    n1 = torch.cross(n, lines)
    n1 /= torch.linalg.norm(n1, axis=1, keepdims=True) + 10e-7

    dist = torch.abs(torch.sum(torch.multiply(pts - pt, n1), axis=1)) 
    _,ii_nn = torch.topk(dist, num_nn + 1, largest=False)

    return ii_nn[1:num_nn + 1]

def get_n_closest_rays_from_point_torch(pt, pts, lines, num_nn):
    num_pts = len(pts)
    
    n = torch.cross(pts-pt,lines)
    n /= torch.linalg.norm(n, axis=1, keepdims=True) + 10e-7
    
    n1 = torch.cross(n, lines)
    n1 /= torch.linalg.norm(n1, axis=1, keepdims=True) + 10e-7

    dist = torch.abs(torch.sum(torch.multiply(pts - pt, n1), axis=1)) 
    _,ii_nn = torch.topk(dist, num_nn, largest=False)

    return ii_nn[:num_nn]

def calc_estimate_from_line(pt_est, line_est, pt_use, line_use):
    n = np.cross(line_est, line_use)  # li, lj 모두에 orthogonal. Distance Vector의 방향
    n /= np.linalg.norm(n) + 10e-7

    n2 = np.cross(n, line_use)
    n2 /= np.linalg.norm(n2) + 10e-7

    est = np.dot((pt_use - pt_est), n2) / (np.dot(line_est, n2) + 10e-7)
    return est

def calc_estimates_from_lines(pt, line, neigh_pts, neigh_lines):
    ests = []
    for i in range(neigh_lines.shape[0]):
        est = calc_estimate_from_line(pt, line, neigh_pts[i, :], neigh_lines[i, :])
        ests.append(est)

    return ests

def find_peak(estimates, num_bins=500):
    # Implement the Kolmogorov-Smirnov and Kuiper's here
    hist_cs = np.zeros(num_bins + 1)
    uni_cs = np.zeros(num_bins + 1)

    uni = np.ones(num_bins) / num_bins
    hist, edges = np.histogram(estimates, bins=num_bins)

    hist_cs[1:] = np.cumsum(hist) / np.sum(hist)
    uni_cs[1:] = np.cumsum(uni) / np.sum(uni)

    min_diff_ind = np.argmin(hist_cs[0:int(0.9 * num_bins)] - uni_cs[0:int(0.9 * num_bins)])
    max_diff_ind = np.argmax(hist_cs[min_diff_ind:-1] - uni_cs[min_diff_ind:-1]) + min_diff_ind
    kuipers_value = hist_cs[max_diff_ind] + uni_cs[min_diff_ind] - hist_cs[min_diff_ind] - uni_cs[max_diff_ind]
    
    in_peak = (estimates < edges[max_diff_ind]) & (estimates > edges[min_diff_ind])
    return in_peak, kuipers_value  # Returns the indices of estimates within peak and kuipers's statistic

def get_peak(estimates, num_bins=500, nro=5):
    est_clean = np.sort(estimates)[nro:-1 * nro]
    kv = 1
    max_kv = 0
    while est_clean.shape[0] > 5 and kv > 0.3:
        in_peak, kv = find_peak(est_clean, num_bins)
        est_clean = est_clean[in_peak]
        if kv > max_kv:
            max_kv = kv
    if (est_clean.shape[0] == 0):
        peak = np.median(estimates)
    else:
        peak = np.median(est_clean)
    return peak, max_kv

def get_peak_kde(estimates, gt_beta, X_MAX, drawGraph, num_bins=500, nro=5):
    THRESHOLD_HEIGHT = 0.001 # Minimum Height between two peaks
    HEIGHT_RANGE = np.linspace(THRESHOLD_HEIGHT, 0.2, 10)

    counts, bins = np.histogram(estimates.flatten(), bins=num_bins) # Graph 보여주기 용
    kde_estimator = stats.gaussian_kde(estimates.flatten(), bw_method=0.07)

    x_axis = np.linspace(-10, X_MAX, 10000)
    K = kde_estimator(x_axis)

    for h in HEIGHT_RANGE[::-1]:
        peaks, _ = find_peaks(K, height=h) # height: Required minimal height of peaks
        idx = heapq.nlargest(2, range(len(K[peaks])), key=K[peaks].__getitem__)
        if len(idx) == 2:
            break

    if len(idx) == 2:
        _a, _b = x_axis[peaks[idx]]

    elif len(K[peaks[idx]]) == 1:
        _a, _b = x_axis[peaks[idx]], x_axis[peaks[idx]] + np.random.uniform(-0.05, 0.05)

    else:
        # TODO More resonable guess
        _a, _b = np.random.uniform(-2, 2), np.random.uniform(-2, 2)

    if abs(_a) > abs(_b):
        _a, _b = _b, _a
    
    norm_coeff = np.max(counts)/(np.max(K) +1e-7) # modified 10.30. jhl
   
    return _a, _b

def estimate_all_pts_one_peak(pts, lines, nns):
    num_pts = pts.shape[0]
    estimates = {}
    estimates_peak = np.zeros([num_pts])

    for i in tqdm(range(num_pts)):
        if isinstance(nns, dict):
            nn = nns[i]
        else:
            nn = nns[i,:]
        estimates[i] = calc_estimates_from_lines(pts[i, :], lines[i, :], pts[nn, :], lines[nn, :])

    print("Finding peaks.")
    for i in range(num_pts):
        estimates_peak[i], _ = get_peak(np.array(estimates[i]))

    return estimates_peak

def estimate_all_pts_two_peaks(pts, lines, nns, gt_beta_B, X_MAX):
    num_pts = pts.shape[0]
    estimates = {}
    estimates_peak1 = np.zeros([num_pts])
    estimates_peak2 = np.zeros([num_pts])
    print("Calculate candidates for estimation")
    for i in tqdm(range(num_pts)):
        if isinstance(nns, dict):
            nn = nns[i]
        else:
            nn = nns[i,:]
        estimates[i] = calc_estimates_from_lines(pts[i, :], lines[i, :], pts[nn, :], lines[nn, :])

    print("Start TPF")
    for i in tqdm(range(num_pts)):
        if i < 7:
            drawGraph = False
        else:
            drawGraph = False
        try:
            estimates_peak1[i], estimates_peak2[i] = get_peak_kde(np.array(estimates[i]), gt_beta_B[i], X_MAX, drawGraph)

        except ValueError:
            print("No estimate(peak) is found", get_peak_kde(np.array(estimates[i]), gt_beta_B[i]), X_MAX, False)
            counts, bins = np.histogram(np.array(estimates[i]).flatten(), bins=500)
            plt.stairs(counts, bins)
            plt.show()
            exit(1)
                    
    return estimates_peak1, estimates_peak2

def line_rejection_percentage(lines_direction, threshold_ratio, center_pts):
    '''
    lines_direction: Direction vectors of 3D lines(rays)
    threshold_ratio: Beta (%)
    center_pts: center points of ray cloud
    '''

    # angles
    angles = []

    # Set default as False
    rejection = np.zeros(len(lines_direction))

    # Select two points
    combinations = list(itertools.combinations(center_pts, 2))
    combinations = np.array(combinations)

    for (center_pt1, center_pt2) in combinations:
        # Normalize baseline vector
        baseline = np.subtract(center_pt2, center_pt1)
        baseline /= np.linalg.norm(baseline)
        lines_direction /= np.linalg.norm(lines_direction, axis=1, keepdims=True)
        print('line rejection center1:', center_pt1, 'center2:', center_pt2, 'baseline:', baseline)
        
        for i, line_direction in enumerate(lines_direction):
            
            # Calculating the angles between two lines using dot products
            angle1 = np.arccos(np.dot(line_direction, baseline))
            angle2 = np.arccos(np.dot(line_direction, -baseline))

            angle1 = np.degrees(angle1)  # Rad to degree
            angle2 = np.degrees(angle2)  # Rad to degree

            angle1 = np.minimum(angle1, (180 - angle1))
            angle2 = np.minimum(angle2, (180 - angle2))

            angle = np.minimum(angle1, angle2)
            angles.append(angle)

    angles = np.array(angles)
    num_filtered_lines = int(len(lines_direction) * threshold_ratio)
    sorted_angles_indices = np.argsort(angles)
    rejected_indices = sorted_angles_indices[:num_filtered_lines] # Get indices of lower angles below threshold

    rejection[rejected_indices] = 1

    print('Rejected line numbers:', len(np.where(rejection == 1)[0]))
    print('Filtered line numbers:', len(np.where(rejection == 0)[0]))

    return rejection

def line_rejection_angle(lines_direction, threshold_angle, anchor_pts):

    print(f'Filtering lines with thrshold {str(threshold_angle)}')

    # Set default as False
    rejection = np.zeros(len(lines_direction))

    # Select two points
    combinations = list(itertools.combinations(anchor_pts, 2))
    combinations = np.array(combinations)

    for (anchor_pt1, anchor_pt2) in combinations:

        # Normalize vector
        anchor_line = np.subtract(anchor_pt2, anchor_pt1)
        anchor_line /= np.linalg.norm(anchor_line)
        lines_direction /= np.linalg.norm(lines_direction, axis=1, keepdims=True)
        print('anchor_line:', anchor_line)
        angles = []
        for i, line_direction in enumerate(lines_direction):
            # Calculating the angles between two lines using dot products
            angle1 = np.arccos(np.dot(line_direction, anchor_line))
            angle2 = np.arccos(np.dot(line_direction, -anchor_line))

            angle1 = np.degrees(angle1) # Rad to degree
            angle2 = np.degrees(angle2) # Rad to degree

            angle1 = np.minimum(angle1, (180-angle1))
            angle2 = np.minimum(angle2, (180-angle2))

            angles.append(angle2)
            # If the angle is smaller than threshold, regard it as parallel
            if angle1 < threshold_angle or angle2 < threshold_angle:
                rejection[i] = 1

    print('Rejected line numbers:', len(np.where(rejection==1)[0]))
    print('Filtered line numbers:', len(np.where(rejection==0)[0]))

    return rejection


def line_rejection_percentage_multi(lines, threshold_ratio, center_pts, pts_to_center_indices):
    '''
    @lines: 3D line directions
    @center_pts: center points (k-means?)
    @pts_to_center_indices: pt -> center mapping
    '''
    rejection = np.zeros(len(lines))
    rejected_indices_total = []
    centers_baselines = []
    
    # Calculate baselines
    # Select two points
    combinations = list(itertools.combinations(center_pts, 2))
    combinations = np.array(combinations)
    print('len(combinations):', len(combinations))
    
    for (center_pt1, center_pt2) in combinations:
        # Normalize baseline vector
        baseline = np.subtract(center_pt2, center_pt1)
        baseline /= np.linalg.norm(baseline)
        centers_baselines.append([center_pt1, center_pt2, baseline])
    
    # Calculate with baseline
    for center1, center2, baseline in centers_baselines:
        
        angles_ith = []
        
        # Indices of center points
        center1_idx = np.where(center1 == center_pts)[0][0]
        center2_idx = np.where(center2 == center_pts)[0][0]
        print('center1 idx:', center1_idx, 'center2 idx:', center2_idx)
        
        lines1_indices = np.where(pts_to_center_indices==center1_idx)[0]
        lines2_indices = np.where(pts_to_center_indices==center2_idx)[0]
        lines_indices = np.union1d(lines1_indices, lines2_indices)
        lines_ith = lines[lines_indices]
        print('len lines1_indices:', len(lines1_indices), 'len lines2_indices:', len(lines2_indices), 'len lines_indices:', len(lines_indices))
        
        for line in lines_ith:
            
            # Check norm of line & baseline
            assert np.linalg.norm(line) <= (1 + (1e-4))
            assert np.linalg.norm(baseline) <= (1 + (1e-4))
    
            cos_angle = np.dot(line, baseline) / (np.linalg.norm(line) * np.linalg.norm(baseline))
            cos_angle = min(1.0, max(cos_angle, -1.0))
            
            angle_tmp = np.degrees(np.arccos(cos_angle))
            angle = np.minimum(angle_tmp, (180 - angle_tmp))
            angles_ith.append(angle)
        
        # Find indices of minimum angles 
        angles_ith = np.array(angles_ith)
        num_filtered_lines = int(len(lines_ith) * threshold_ratio)
        sorted_angles_indices = np.argsort(angles_ith)
        ith_rejected_indices_tmp = sorted_angles_indices[:num_filtered_lines] # Get indices of lower angles below threshold
        ith_rejected_indices = lines_indices[ith_rejected_indices_tmp]
        rejected_indices_total.append(ith_rejected_indices)
        
        print('len lines_ith:', len(lines_ith))
        print('len ith_rejected_indices:', len(ith_rejected_indices))
        
    print('#'*50)
    rejected_indices_total = np.array(list(itertools.chain.from_iterable(rejected_indices_total)))
    rejected_indices_total = np.unique(rejected_indices_total)
    print('total lines:', lines.shape[0])
    print('rejected_indices_total shape:', rejected_indices_total.shape[0])
    print('#'*50)
    
    # Set
    rejection[rejected_indices_total] = 1
    
    return rejection

def calculate_closest_p2p(pts):

    num_nn_p2p = 500 # used number of neighbors for reconstruction
    p2p_lst = []
    num_pts = len(pts)

    # Build ckDTree for all points
    kdtree = cKDTree(pts)
    for i in tqdm(range(num_pts)):
        query_point = pts[i]
        # Get closest point-point distances and indices with the number of num_nn_p2p
        distances, indices = kdtree.query(query_point, k=num_nn_p2p+1)
        p2p_lst.append(indices[1:]) # Exclude containing by itself

    p2p_lst = np.array(p2p_lst, dtype=np.int32)
    print('p2p_lst shape:', p2p_lst.shape)
    return p2p_lst
