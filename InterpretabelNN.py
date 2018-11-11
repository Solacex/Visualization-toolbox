import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from numpy import linalg as LA
feat_file = scipy.io.loadmat('softmax_raw_result.mat')
attr_file = scipy.io.loadmat('test_attribute.mat')

ga_feat = feat_file['gallery_f']
qu_feat = feat_file['query_f']

attr = attr_file['attribute']
a_key = attr_file['key']


attr = attr[1:,:]
ga_label = feat_file['gallery_label'][0]
qu_label = feat_file['query_label'][0]


num_img = ga_feat.shape[0]
num_attr  = attr.shape[0]
num_dim = ga_feat.shape[1]

junk = np.zeros(num_attr)


dic_label={}
for s in range(len(ga_label)):
    cu_label = ga_label[s]
    if cu_label==-1:
        continue
    if cu_label not in dic_label:
        dic_label[cu_label] = len(dic_label)

washed_index =np.where(ga_label!=-1)
ga_feat = np.squeeze(ga_feat[washed_index,:])


attr_img = np.zeros((num_attr,num_img), dtype=float)
#attr_img = np.zeros(num_attr, dtype=float)

for i in range(num_img):
    label = ga_label[i]

    if label==-1:
        continue
    else:
        cu_label = dic_label[label]
        attr_img[:,i] = attr[:,cu_label]
            #attr_img = np.concatenate((attr_img,attr[:,cu_label]), axis=1)

attr_img = np.squeeze(attr_img[:,washed_index])
print('junk sample cleaned', attr_img.shape, ga_feat.shape)
num_img = ga_feat.shape[0]
num_attr  = attr.shape[0]
num_dim = ga_feat.shape[1]



count_by_attr = np.sum(attr_img,axis=1, keepdims=True)
ratio_for_attr = 1 - count_by_attr/num_img
mean_ratio = int(np.mean(ratio_for_attr)*100)
print('average ratio for attribute in images:', mean_ratio)
count_by_attr = np.repeat(count_by_attr,num_dim, axis=1)

count_by_unit = np.sum(ga_feat,axis=0, keepdims=True)
count_by_unit = np.repeat(count_by_unit,num_attr, axis=0)


'''
percen = np.percentile(ga_feat,90, axis=1,keepdims=True, interpolation='higher')
for i in range(ga_feat.shape[0]):
  #  print(np.where(ga_feat[i,:]>=percen[i]))
    ga_feat[i,np.where(ga_feat[i,:]>=percen[i])]=1
    ga_feat[i,np.where(ga_feat[i,:]<percen[i])]=0
'''
percen = np.percentile(ga_feat,mean_ratio, axis=0, interpolation='higher')
for i in range(num_dim):
    ga_feat[np.where(ga_feat[:,i] >=percen[i]),i]=1
    ga_feat[np.where(ga_feat[:,i] < percen[i]),i]=0


print(np.sum(ga_feat, axis=0)[:10])

relation = np.matmul(attr_img, ga_feat)
score = np.divide(relation, count_by_attr)
#score = score-score_in
score2 = np.divide(relation, count_by_unit)



location = np.argmax(score2)
dim1 = int(location/num_dim)
dim2 =location%num_dim - 1

print('Attribute:', a_key[dim1], ' Unit:', dim2)

max_score = score[dim1][dim2]
max_score2 = score2[dim1][dim2]

print(max_score*100,'% of attributes')
print(max_score2*100,'% of units')

###################################################
#Visualization
###################################################

tov_attr  = np.squeeze(attr_img[dim1,:])
tov_unit = np.squeeze(ga_feat[:,dim2])*0.5
#tov_norm = LA.norm(tov_unit, ord=2)
#tov_unit = tov_unit/tov_norm
#tov_unit*=6
fig  = plt.figure()
ax = fig.add_subplot(1,1,1)
axis_x  = np.arange(num_img)+1

ax.bar(axis_x, tov_attr, label='attribute', color='g', alpha=0.5)

ax.bar(axis_x, tov_unit, label='unit', color='r', alpha=0.5)
plt.legend(loc = 'upper right')
plt.show()




