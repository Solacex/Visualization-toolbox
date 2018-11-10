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

#attr[0,:]=0
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

#np.save('ga_feat.npy', ga_feat)
#np.save('attr_img.npy', attr_img)
ga_feat = np.load('ga_feat.npy')
attr_img = np.load('attr_img.npy')
num_img = ga_feat.shape[0]
num_attr  = attr.shape[0]
num_dim = ga_feat.shape[1]

attr_img_in = 1 - attr_img # in means inverse


count_by_attr = np.sum(attr_img,axis=1, keepdims=True)
count_by_attr = np.repeat(count_by_attr,num_dim, axis=1)
count_by_attr_in = np.sum(attr_img_in,axis=1, keepdims=True)
count_by_attr_in = np.repeat(count_by_attr_in,num_dim, axis=1)
#print(count_by_attr)
print(a_key)
#print(ga_feat.shape)
#attr = torch.Tensor(attr_img).float()
#g_feat = torch.Tensor(ga_feat)

percen = np.percentile(ga_feat,90, axis=1,keepdims=True, interpolation='higher',)
#print(percen[0])
for i in range(ga_feat.shape[0]):
  #  print(np.where(ga_feat[i,:]>=percen[i]))
    ga_feat[i,np.where(ga_feat[i,:]>=percen[i])]=0.5
    ga_feat[i,np.where(ga_feat[i,:]<percen[i])]=0

print(np.sum(ga_feat, axis=1)[:10])

score = np.divide(np.matmul(attr_img, ga_feat), count_by_attr)
score_in = np.divide(np.matmul(attr_img_in, ga_feat), count_by_attr_in)
#score = score-score_in
location = np.argmax(score)
print(score.shape)
dim1 = int(location/num_dim),
dim2 =location%num_dim - 1
print(location, dim1, dim2)
###################################################
#Visualization
###################################################

tov_attr  = np.squeeze(attr_img[dim1,:])
tov_unit = np.squeeze(ga_feat[:,dim2])
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

