from PIL import Image
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage.io import imread
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.transform import resize
import numpy as np
import cv2
import pickle
from torchvision.transforms import ToPILImage
import torch

import warnings
warnings.filterwarnings('ignore')

import threading

data = pickle.load(open('/home/ukjung18/pytorch-classification/traindata/data_dict.pickle', 'rb'))
train_set, test_set, val_set = data['train'], data['test'], data['val']

def preprocess(i):
    for path, label in train_set[3500*i:3500+3500*i]:
        name = path.split('/')[-1]

        clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize=(4, 4))
        image = imread(path)
        image = Image.fromarray((image*255).astype(np.uint8))

        image = resize(image, (512, 512))
        img = ToPILImage()(image)
        img.save('/home/ukjung18/pytorch-classification/traindata/train_resize/'+name)
    print('Train done: ', i)

if __name__ == '__main__':

    threads = []
    for i in range(10):
        t = threading.Thread(target=preprocess, args=[i])
        t.start()
        threads.append(t)
        
    for thread in threads:
        thread.join()
    # test()

# with open(path, 'rb') as f:
#     # img = Image.open(f).convert('RGB')
#     # r, g, b = img.split()
#     # r= r.point(lambda i: i*0)
#     # b= b.point(lambda i: i*0)
#     # green = Image.merge('RGB', (r,g,b))

#     return img


# clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize=(16, 16))
# image = imread(path)
# image = Image.fromarray((image*255).astype(np.uint8))

# image_np = np.array(image)
# opencv_img=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
# opencv_img=cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
# r,g,b = cv2.split(opencv_img)

# r = clahe.apply(r)
# g = clahe.apply(g)
# b = clahe.apply(b)
        
# image = g
# img_array=np.array(image)
# image=np.repeat(img_array[:,:,np.newaxis],3,-1) #3channel
# r_array = np.array(r)
# g_array = np.array(g)
# b_array = np.array(b)
# image = np.concatenate((r_array[:,:,np.newaxis], g_array[:,:,np.newaxis], b_array[:,:,np.newaxis]), -1)
# img=Image.fromarray((image).astype(np.uint8))