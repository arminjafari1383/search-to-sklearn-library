import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
shape_img = (4,4,1)
mask = np.zeros(shape=shape_img, dtype=bool)
mask[[1,2],[1,2],:] = True
graph = grid_to_graph(*shape_img,mask=mask)
print(graph)

