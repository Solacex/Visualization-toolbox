import matplotlib.cm as cm
import numpy 
import scipy.io
import sklearn
import sys 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from PIL import ImageDraw
feat_path = sys.argv[1]
#feat = sys.argv[2]


img_path = '/home/liguangrui3/data/hw36w/select_plus_new/gallery/' 


def combine_img(fig, img_list, lb_list):
    w = 128
    h = 256
    ori_ls = img_list
    num_img  = len(ori_ls)
    i = 0
    final_img = []
    for paths in ori_ls:        
        cu_num = len(paths)
        new_w = w*cu_num
        tmp_list = []
        for i in range(cu_num):
            t_img = (Image.open(img_path+'/'+paths[i][:30]+'/'+paths[i])).resize((w,h))
            draw = ImageDraw.Draw(t_img)
            draw.text((0,0), str(i+1), fill=(0,0,0))
            tmp_list.append(t_img)
        target = Image.new('RGB', (new_w, h))
        left = 0
        right = w
        for img in tmp_list:
            target.paste(img, (left, 0, right, h))
            left  += w
            right += w
        final_img.append(target)
    for i in range(num_img):
        img = final_img[i]
         
        fig.add_subplot(int(num_img/2),2,i+1)
        plt.title(lb_list[i])
        plt.imshow(img)
    return final_img

result = scipy.io.loadmat(feat_path)

gallery_feature = result['gallery_f']
gallery_label = result['gallery_label']

tsne = TSNE(n_components = 2)
data = gallery_feature[:1000]
lbs = gallery_label[:1000]
lb=[]
for m in lbs:
    lb.append(m[:30])

#print(lb[:10])
lb = np.array(lb)

lb_list = np.unique(lb)
cls_num = len(lb_list)
print('Number of classes for training:', len(lb_list))

new_data = tsne.fit_transform(data)
dif_list=[]
path_list = []
num_cls = 40
for i in range(num_cls):
     dif_list.append(new_data[np.where(lb==lb_list[i]),:])
     path_list.append(lbs[np.where(lb==lb_list[i])])

clr=['b','g','r', 'c', 'm', 'y', 'k']
shape = [ ',', 'o', 'v', '^', '<', '>','D']

#img_plt = plt.figure()

#combine_img(img_plt, path_list, lb_list)
fig = plt.figure(figsize=(30,20))
ax = plt.subplot()
for i in range(num_cls):
    mark_tmp1 = int(i/7)
    mark_tmp2 = int(i%7)
    cu = np.squeeze(dif_list[i])
    ax.scatter(cu[:,0], cu[:,1],verts=np.arange(len(cu)), label=str(lb_list[i]), marker=shape[mark_tmp1],c=clr[mark_tmp2], alpha=0.6)
    for m in range(len(cu)):
        ax.annotate(m, (cu[m,0], cu[m,1]))
plt.legend()
#plt.scatter(new_data[:,0],new_data[:,1], alpha=0.6)

save_name = feat_path.split('.')[-2] + '.png'
print(save_name)
plt.savefig('../vis/' + save_name)
plt.show() 



#final = {'feature': new_data}
#scipy.io.savemat(path.split('.')[0]+'_vis.mat', final)


