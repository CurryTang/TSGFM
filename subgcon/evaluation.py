import numpy as np
from sklearn import metrics
import sklearn.metrics as Metrics
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict

def process_set(sets):
    comms = []
    labels = set(sets)
    for label in labels:
        comm = set(np.where(sets == label)[0])
        comms.append(comm)
    return comms 

def NMI_helper(found_sets, GT_sets):
    NMI_result = metrics.normalized_mutual_info_score(GT_sets,found_sets)
    return NMI_result

def F_score_helper(GT, found, common_elements):
    len_common = len(common_elements)
    precision = float(len_common)/len(found)
    if precision == 0:
        return 0

    recall = float(len_common)/len(GT)
    if recall == 0:
        return 0
    return (2*precision*recall)/(precision+recall)


def cal_F_score_helper(found_sets, GT_sets):
    d1 = {} #best match for an extracted community
    d2 = {} #best match for a known community

    for i in range(len(GT_sets)):
        gt = GT_sets[i]
        f_max = 0

        for j in range(len(found_sets)):
            f = found_sets[j]

            common_elements = gt.intersection(f)
            if len(common_elements) == 0:
                temp = 0
            else:
                temp = F_score_helper(gt, f, common_elements)

            f_max = max(f_max,temp)

            d1[j] = max(d1.get(j,0),temp)

        d2[i] = f_max

    return d1, d2


def cal_F_score(found_sets, GT_sets, verbose=False):
    found_sets = process_set(found_sets)
    GT_sets = process_set(GT_sets)
    d1,d2 = cal_F_score_helper(found_sets, GT_sets)

    if d1 == None:
        return [0]*6

    vals1 = sum(d1.values())/len(d1)
    vals2 = sum(d2.values())/len(d2)
    f_score = vals1 + vals2
    f_score /= 2
    f_score = round(f_score,4)
    vals1 = round(vals1,4)
    vals2 = round(vals2,4)

    return f_score, vals1, vals2

def matched(true,pred):
    max_idx = max(max(true),max(pred))
    cm = Metrics.confusion_matrix(true,pred,labels=np.arange(0,max_idx+1))
    shifted_mat = np.zeros((cm.shape[0]*2,cm.shape[0] * 2))
    shifted_mat[:cm.shape[0],cm.shape[0]:] = cm
    g = nx.from_numpy_matrix(shifted_mat)
    match = nx.max_weight_matching(g)
    unmatched = set(np.arange(0,cm.shape[0]))
    label_map = {}
    for m in match:
        p,t = max(m),min(m)
        unmatched.remove(t)
        label_map[p] = t
    unmatched = list(unmatched)
    for i in range(cm.shape[0],cm.shape[0]*2):
        if not i in label_map:
            label_map[i] = unmatched[-1]
            unmatched.pop()
   
    for i in range(len(pred)):
        pred[i] = label_map[pred[i]+cm.shape[0]]
    return pred

def matched_cm(true,pred):
    max_idx = max(max(true),max(pred))
    pred = matched(true,pred,labels=np.arange(0,max_idx+1))
    cm = Metrics.confusion_matrix(true,pred)
    return cm

def matched_ac(pred, true):
    pred = matched(true,pred)
    ac = Metrics.accuracy_score(true,pred)
    return ac
#score = cal_F_score(found_comms, groundtruth_comms)
#print ("f1:",score[0])