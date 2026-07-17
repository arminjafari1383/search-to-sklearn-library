import numpy as np
from sklearn.feature_extraction.image import img_to_graph

img = np.array([[0,0],[0,1]])
y1 = img_to_graph(img,return_as=np.ndarray)
print(y1)

