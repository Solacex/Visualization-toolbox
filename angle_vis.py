import sys
import os
import numpy as np
import time
import math
import multiprocessing as mtp
from numpy import *
import logging
import argparse
import scipy.io

import scipy.io
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from PIL import ImageDraw

mat_name= sys.argv[1]
if 'select' in mat_name:
    img_path = '/home/liguangrui3/data/hw36w/select_plus_new/gallery/'
elif 'Actor' in mat_name:
    img_path = '/home/liguangrui3/data/hw36w/f-Ped-Actor-test/gallery/'
elif 'cs' in mat_name:
    img_path = '/home/liguangrui3/data/hw36w/kf-ped-5on1-cs/gallery/'
elif 'kf' in mat_name:
    img_path = '/home/liguangrui3/data/hw36w/kf-ped-5on1/gallery/'
else:
    img_path = '/home/liguangrui3/data/hw36w/k-ped-test/gallery/'

def combine_img(fig, top_s, top_dis, name):
    w = 128
    h = 256
    
    num_img  = len(top_s)
    i = 0
    final_img = []
    for ls in [top_s, top_dis]:
        tmp_list = []
        for tup in ls:
            
            new_w = w*num_img
            new_h = h*2
            for i in tup:
                t_img = (Image.open(img_path+i[:30]+'/'+i)).resize((w,h))
               # draw = ImageDraw.Draw(t_img)
               # draw.text((0,0), str(i+1), fill=(0,0,0))
                tmp_list.append(t_img)
        target = Image.new('RGB', (new_w, new_h))
        left = 0
        right = w
        top = 0
        bot = h
        for i in range(len(tmp_list)):
            target.paste(tmp_list[i], (left, top, right, bot))
            if top ==0:
                top += h
                bot += h
            else:
                top = 0
                bot = h
                left += w
                right += w
        final_img.append(target)
    for i in range(len(final_img)):
        img = final_img[i]

        fig.add_subplot(2,1,i+1)
        #plt.title(lb_list[i])
        plt.imshow(img)
    plt.savefig('./vis/angle/' + name)
    return final_img

#######################################################################
# Evaluate
def evaluate(qf,ql,qp,gf,gl,gp):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    s_inde = np.where(gl==ql)
    dis_inde = np.where(gl!=ql)
    #print(s_inde, dis_inde) 
    s_ang = score[s_inde]
    dis_ang = score[dis_inde]

    s_path = [(qp, gp[i]) for i in s_inde[0] ]
    dis_path = [(qp, gp[i]) for i in dis_inde[0]]
 
#    s_ang = np.array(s_ang)
 #   dis_ang =  np.array(dis_ang)
    s_ang = list(s_ang)
    dis_ang =  list(dis_ang)
    return s_ang, dis_ang, s_path, dis_path



######################################################################
mat_name= sys.argv[1]
print(mat_name)
result = scipy.io.loadmat(mat_name)
query_feature = torch.FloatTensor(result['query_f'])
query_labels = result['query_label']
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_labels = result['gallery_label']
query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print('query shape: ', query_feature.shape, 'gallery shape: ', gallery_feature.shape)
ap = 0.0

#print(query_label)
gl=[]
ql=[]
s = np.array([0])
dis= np.array([0])

for m in gallery_labels:
    gl.append( m[:30])

for j in query_labels:
    ql.append(j[:30])

gl =  np.array(gl)
ql = np.array(ql)
s=[]
dis=[]
s_path = []
dis_path = []
for i in range(len(ql)):
    sa, disa, sp, disp = evaluate(query_feature[i],ql[i],query_labels[i], gallery_feature, gl, gallery_labels)
    s.extend(sa)
    dis.extend(disa)
    s_path.extend(sp)
    dis_path.extend(disp)
    #print(s)
    #if i ==5:
     #   break
   # print(len(s))
    #s = np.concatenate((s, sa), axis=0)
    #dis = np.concatenate((dis, disa), axis=0)
    #print(s[0].shape, dis[0].shape)
    #break
#print(sa.shape,disa.shape)
s = np.array(s)
dis = np.array(dis)

s = np.clip(s,-1,1)
dis = np.clip(dis, -1, 1)
s_angle = np.arccos(s)
dis_angle = np.arccos(dis)
s_inde = np.argsort(s_angle)[-30:]
dis_inde = np.argsort(dis_angle)[:30]
top_sp = [s_path[i] for i in s_inde]
top_disp = [dis_path[i] for i in dis_inde]

ori_name = mat_name.split('/')[-1]

fig2 = plt.figure(figsize=(30,20))
print(top_disp)
save_name2 = ori_name.split('.')[0]+'_hard_sample.png'
combine_img(fig2, top_sp, top_disp, save_name2)
#dis_angle = dis_angle[::1000]
#s_angle = s_angle(np.where(s_angle!=0))

pos_max = np.max(s_angle)
neg_min = np.min(dis_angle)
margin = neg_min - pos_max 

anot = 'max_angle(pos): '+str(pos_max)+' min_angle(neg):'+str(neg_min)+' margin: '+str(margin)

print(s_angle.shape, dis_angle.shape) 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#bx = fig.add_subplot(1,2,2)
bins = np.linspace(0,3.2,80)


ax.hist(s_angle, bins, normed=1,label='positive pairs angle', color='g', alpha=0.5)
ax.hist(dis_angle, bins, normed=1,label='negative pairs angle', color='r', alpha=0.5)
plt.text(0,6, anot)
plt.legend(loc = 'upper right')
ori_name = mat_name.split('/')[-1]
save_name = ori_name.split('.')[0]+'_angle.png'

plt.savefig('./vis/angle/' + save_name)
plt.show()
