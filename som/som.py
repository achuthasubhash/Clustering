# Self Organizing Map
#reduce dim to visualisation
#here weights are rep intially, then find dis,less dis call as best matching unit,
#party is real,som is classify based on answer to cluster(did by som) red-yes
#larger mid closer to white

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values #use cus ID to identify ID in result
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
#x,y is dimension of som
#input len is feature here 15 because to catch faruad so use ID also,sigma is radius ,l_r is how much weight iterate for each iteration
#higher l_r fastely get O/P
som.random_weights_init(X) #ransom weights
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone() #get blank window 
pcolor(som.distance_map().T) #transpose of dis vector
colorbar() #legend highest distance then 1
markers = ['o', 's']
colors = ['r', 'g'] #red circle not approved
for i, x in enumerate(X): #i is 1,2,3...,x is vector of values
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5, #centre
         markers[y[i]], #gives approved or  not
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',#inside color
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X) #gives list of customer of winning node
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0) #0 is vertical
frauds = sc.inverse_transform(frauds)