import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def evaluation(probe, gallery, valid = False):
    probe_feature, seq_id, _, _ = probe
    gallery_feature, _, _, gait_id = gallery
    seq_id = np.array(seq_id)
    gait_id = np.array(gait_id)

    dist = cuda_dist(probe_feature, gallery_feature)
    dist = dist.cpu().numpy()
    idx = np.argmin(dist, axis=1)
    seq_gait_id = gait_id[idx]
    s = []
    dic = {}
    for i in range(len(seq_id)):
        s.append([seq_id[i], seq_gait_id[i]])
        dic[seq_id[i]] = seq_gait_id[i]
        
    if valid:
        f = open('../dataset/labels.pkl', 'rb')
        labels = pickle.load(f)
        f.close()
        
        acc = 0
        for i, id_ in enumerate(labels[0]):
            if id_ in dic.keys():
                if dic[id_]==labels[1][i]:
                    acc += 1
        return acc / len(labels[0])  
    else:
        print(s)
        print(s[0])
        print("videoID", s[0][0], "subjectID", s[0][1])
        # return s[0][0], s[0][1]
        return s[0][1]
        # submission = pd.DataFrame(s, columns=['videoID', 'subjectID'])
        # submission.to_csv('submission.csv', index=False)
