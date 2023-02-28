import pathlib
import numpy as np
import cv2 
import os


       
def make_dataset(paths, scale):
    """
    Creates input and ground-truth.
    """
    for p in paths:
        input_img_path = p.decode()

        # 自动根据输入图像路径匹配标签图像路径
        target_img_path = input_img_path.replace('inputs','targets')
        
        # read
        im = cv2.imread(input_img_path, 3).astype(np.float32)
        # convert to YCrCb (cv2 reads images in BGR!), and normalize
        im_ycc_lr = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb) / 255.0

        im = cv2.imread(target_img_path, 3).astype(np.float32)
        # convert to YCrCb (cv2 reads images in BGR!), and normalize
        im_ycc_tag = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb) / 255.0

    
        # make current image divisible by scale (because current image is the HR image)
        #im_ycc_hr = cv2.resize(im_ycc_tag, (int(im_ycc_lr.shape[1] * scale), int(im_ycc_lr.shape[0] * scale)), 
        #                  interpolation=cv2.INTER_CUBIC)

        #print(input_img_path.split('\\')[-1])
        #print(im_ycc_lr.shape[:2],'--->',im_ycc_tag.shape[:2],'expect--->',im_ycc_hr.shape[:2])
        
        # 检查输入输出样本对尺寸是否规范
        if (im_ycc_lr.shape[0]*scale ==  im_ycc_tag.shape[0]) and (im_ycc_lr.shape[1]*scale ==  im_ycc_tag.shape[1]):
            pass
        else: print('please check the size about: ',input_img_path)

        # only work on the luminance channel Y
        lr = np.expand_dims(im_ycc_lr[:,:,0], axis=2)
        hr = np.expand_dims(im_ycc_tag[:,:,0], axis=2)
        
        yield lr, hr

def make_train_dataset(paths, scale):
    """
    Python generator-style dataset. Creates low-res and corresponding high-res patches.
    """
    # set lr and hr sizes
    size_lr = 10
    if(scale == 3):
        size_lr = 7
    elif(scale == 4):
        size_lr = 6
    size_hr = size_lr * scale
    
    for p in paths:
        # read 
        im = cv2.imread(p.decode(), 3).astype(np.float32)
        
        # convert to YCrCb (cv2 reads images in BGR!), and normalize
        im_ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb) / 255.0

        # -- Creating LR and HR images
        # make current image divisible by scale (because current image is the HR image)
        im_ycc_hr = im_ycc[0:(im_ycc.shape[0] - (im_ycc.shape[0] % scale)),
                           0:(im_ycc.shape[1] - (im_ycc.shape[1] % scale)), :]
        im_ycc_lr = cv2.resize(im_ycc_hr, (int(im_ycc_hr.shape[1] / scale), int(im_ycc_hr.shape[0] / scale)), 
                           interpolation=cv2.INTER_CUBIC)
        
        # only work on the luminance channel Y
        lr = im_ycc_lr[:,:,0]
        hr = im_ycc_hr[:,:,0]
        
        numx = int(lr.shape[0] / size_lr)
        numy = int(lr.shape[1] / size_lr)
        
        for i in range(0, numx):
            startx = i * size_lr
            endx = (i * size_lr) + size_lr
            
            startx_hr = i * size_hr
            endx_hr = (i * size_hr) + size_hr
            
            for j in range(0, numy):
                starty = j * size_lr
                endy = (j * size_lr) + size_lr
                starty_hr = j * size_hr
                endy_hr = (j * size_hr) + size_hr

                crop_lr = lr[startx:endx, starty:endy]
                crop_hr = hr[startx_hr:endx_hr, starty_hr:endy_hr]
        
                x = crop_lr.reshape((size_lr, size_lr, 1))
                y = crop_hr.reshape((size_hr, size_hr, 1))
                yield x, y

def make_val_dataset(paths, scale):
    """
    Python generator-style dataset for the validation set. Creates input and ground-truth.
    """
    for p in paths:
        # read
        im = cv2.imread(p.decode(), 3).astype(np.float32)
        # convert to YCrCb (cv2 reads images in BGR!), and normalize
        im_ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb) / 255.0
        
        # make current image divisible by scale (because current image is the HR image)
        im_ycc_hr = im_ycc[0:(im_ycc.shape[0] - (im_ycc.shape[0] % scale)),
                           0:(im_ycc.shape[1] - (im_ycc.shape[1] % scale)), :]
        im_ycc_lr = cv2.resize(im_ycc_hr, (int(im_ycc_hr.shape[1] / scale), int(im_ycc_hr.shape[0] / scale)), 
                           interpolation=cv2.INTER_CUBIC)
        
        # only work on the luminance channel Y
        lr = np.expand_dims(im_ycc_lr[:,:,0], axis=2)
        hr = np.expand_dims(im_ycc_hr[:,:,0], axis=2)
        
        
        yield lr, hr

def getpaths(path):
    """
    Get all image paths from folder 'path'
    """
    data = pathlib.Path(path)
    all_image_paths = list(data.glob('*'))
    all_image_paths = [str(p) for p in all_image_paths]
    return all_image_paths



def downscale(img,scale):
    """
    将原图缩小scale倍
    """
    (w, h) = img.shape[:2]
    img_hat = cv2.resize(img, dsize=(int(h/scale),int(w/scale)), interpolation=cv2.INTER_CUBIC)
    
    return img_hat

def upscale(img,scale):
    """
    将原图缩小scale倍
    """
    (w, h) = img.shape[:2]
    img_hat = cv2.resize(img, dsize=(int(h*scale),int(w*scale)), interpolation=cv2.INTER_CUBIC)
    
    return img_hat

if __name__ == '__main__':
    make_dataset('traindir/inputs', 'traindir/targets')