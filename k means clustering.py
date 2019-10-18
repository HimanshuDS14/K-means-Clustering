import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

x = -2* np.random.rand(100,2)
y = 1+2*np.random.rand(50,2)

x[50:100 , :] = y

plt.scatter(x[:,0] , x[:,1] , s = 100)
plt.show()

cluster= KMeans(n_clusters=2)
cluster.fit(x)

#find the centroid
cent = cluster.cluster_centers_
print(cent)



#plot data with centroid
plt.scatter(x[:,0] , x[:,1] , s = 100)
plt.scatter(cent[0][0] , cent[0][1] , color = "red" , marker="*" , s = 100)
plt.scatter(cent[1][0] , cent[1][1] , color = "green" , marker="*" , s = 100)
plt.show()

print(cluster.labels_)

#prediction
z = np.array([3.0 , 3.0])
z = z.reshape(1 ,-1)

print(cluster.predict(z))