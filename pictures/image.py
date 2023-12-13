# for all image in this directory let's make every pixel tkat are close to the white without behind exactly white to be 10% darker ( since they may be a little bit blue the will be a little bit more blue)
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.cm as cm


def get_image(path):
    img = Image.open(path)
    return img

def get_image_array(path):
    img = get_image(path)
    return np.array(img)

# for all image in this directory let's make every pixel tkat are close to the white without behind exactly white to be 10% darker ( since they may be a little bit blue the will be a little bit more blue)
def make_white_darker(path):
    img = get_image_array(path)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 254 >img[i][j][0] > 240 and 254 >img[i][j][1] > 240 and 254 >img[i][j][2] > 240:
                img[i][j][0] = img[i][j][0] * 0.9
                img[i][j][1] = img[i][j][1] * 0.9
                img[i][j][2] = img[i][j][2] * 0.9
    return img

# for all image in this directory let's make every pixel tkat are close to the white without behind exactly white to be 10% more dark blue 
def make_white_darker_blue(path):
    img = get_image_array(path)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 255 >img[i][j][0] > 230 and 255 >img[i][j][1] > 230 and 255 >img[i][j][2] > 230:
                img[i][j][0] = img[i][j][0] * 0.9
                img[i][j][1] = img[i][j][1] * 0.9
                img[i][j][2] = img[i][j][2] * 1
    return img

if __name__ == "__main__":
    print("hello")
    path = os.getcwd()
    path = path + "/pictures"
    for filename in os.listdir(path):
        print(filename)
        if filename.endswith(".png"):
            img = make_white_darker_blue (path + "/" + filename)
            #save the image
            plt.imsave(path + "/" + filename, img)
            continue
        else:
            continue
    
    print("end")