import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# reference : 
# https://www.kaggle.com/iafoss/image-preprocessing-128x128
# https://www.kaggle.com/phoenix9032/pytorch-efficientnet-starter-code
# https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch


# param img : np.array 2x2
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


# param img : np.array 2x2
# return img : np.array 2x2
def crop_padding(img0, width, height, pad=16):
    # crop a box around pixels large than the threshold 
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < width - 13) else width
    ymax = ymax + 10 if (ymax < height - 10) else height
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    #img[img < 28] = 0
    lx = xmax - xmin
    ly = ymax - ymin
    l = max(lx, ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    # padding
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return img


# param df : pandas dataframe
def Resize(df, width, height, size=128):
    resized = {}
    df = df.set_index('image_id')
    for i in tqdm(range(len(df))):
        img0 = 255 - df.loc[df.index[i]].values.reshape(137, 236).astype(np.uint8)
        # normalize each image by its max val
        img1 = (img0 * (255.0 / img0.max())).astype(np.uint8)
        img2 = crop_padding(img1, width, height)
        img3 = cv2.resize(img2, (size, size))
        resized[df.index[i]] = img3.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized


# param filename : str
# return np.array
def read_feathers(filenames, size):
    dfs = [pd.read_feather(f) for f in filenames]
    images = [df.iloc[:, 1:].values.reshape(-1, 1, size, size) for df in dfs]
    del dfs
    gc.collect()
    return np.concatenate(images, axis=0)


# param filename : str
# return np.array
def read_parquets(filenames, width, height, size):
    dfs = [pd.read_parquet(f) for f in filenames]
    resized_dfs = [Resize(df, width, height, size) for df in dfs]
    images = [df.iloc[:, 1:].values.reshape(-1, 1, size, size) for df in resized_dfs]
    del dfs
    gc.collect()
    return np.concatenate(images, axis=0)


