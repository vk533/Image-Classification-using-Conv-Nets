from PIL import Image
from PIL import ImageFilter
import numpy as np

#open the image file and convert to matrix
fname = '0001.png'
img = Image.open(fname)
print(img.mode)#Channel 
print(img.size)#Width, Height
img_array = np.array(img)#Height, Width, Channel
img.show()#View image
#print(img_array)

img2 = img.convert('L')
img2.show()#View image
print(img2.mode)#Channel 
print(img2.size)#Width, Height
img_array = np.array(img2)#Height, Width, Channel
print(img_array)

for i in range(0, len(img_array)):
	print(img_array[i])

img3 = img2.filter(ImageFilter.FIND_EDGES)
img3.show()

img3 = img2.filter(ImageFilter.Kernel((3,3),(-1, -1, -1, -1, 8, -1, -1, -1, -1)))
img3.show()

def img2vec(fname=None, flatten=False):
    img = Image.open(fname)
    img_array = np.array(img)
    if flatten:
        return img_array.flatten()
    else:
        return img_array
    
    
#fname = '0001.png'    
#a = img2vec(fname)
#print(a.shape)
#print(a)
