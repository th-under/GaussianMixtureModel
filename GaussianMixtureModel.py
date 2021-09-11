# -*- coding: utf-8 -*-
#!/usr/bin/python
# © 2021: th-under



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.mixture import GaussianMixture


def gauss_function(x, amplitude, x_mean, sd):
    return amplitude * np.exp(-(x - x_mean) ** 2. / (2. * sd ** 2.))




########################  example w/o outliers  ###############################


filename = './data/mocap_data_exc.csv'

csvdata = np.genfromtxt(filename,skip_header=3,delimiter=',',usecols=range(10,19), unpack=True)

ac = csvdata.reshape(3,3,-1) # 3 markers, 3 coords (x,y,z), ~2000 frames

dist_1 = np.linalg.norm(ac[0]-ac[1],axis=0)

pd.DataFrame(dist_1).hist(bins=100)
plt.xlabel('marker distance (mm)')
plt.ylabel('number of data points')
plt.title('marker 1 - marker 2')

# overlay best fit gaussian distribution
x_vals = np.arange(18.3,20.3,0.01)
y_vals = gauss_function(x_vals, amplitude=77, x_mean=19.18, sd=0.208) # values determined in advance

plt.plot(x_vals, y_vals)



########################  example with outliers  ##############################


csvdata = np.genfromtxt(filename,skip_header=3,delimiter=',',usecols=range(1,10), unpack=True)

ac = csvdata.reshape(3,3,-1)

        
# distance over time     

dist_1 = np.linalg.norm(ac[0]-ac[1],axis=0)
times = np.arange(0, csvdata.shape[1]/120, 1/120)

plt.figure()
plt.scatter(times, dist_1, s=2)
plt.xlabel('measuring time (s)')
plt.ylabel('distance (mm)')
plt.title('marker 1 - marker 2')
plt.show()


# non-normal distributed histogram

pd.DataFrame(dist_1).hist(bins=100, density=True)
plt.xlabel('marker distance (mm)')
plt.ylabel('frequency')
plt.title('another marker 1 - marker 2')


gmm = GaussianMixture(n_components=2, covariance_type="full", tol=0.001)
gmm.fit(X=dist_1.reshape(-1,1))

gmm_mean1 = gmm.means_[0,0]
gmm_mean2 = gmm.means_[1,0]

gmm_sd1 = np.sqrt(gmm.covariances_[0,0,0])
gmm_sd2 = np.sqrt(gmm.covariances_[1,0,0])


# likelihood, to which cluster a certain data point belongs to
x_vals = np.linspace(min(dist_1), max(dist_1), len(dist_1))
probabilities = gmm.predict_proba(x_vals.reshape(-1, 1))

# envelope curve
pdf_sum = np.exp(gmm.score_samples(x_vals.reshape(-1, 1)))
plt.plot(x_vals, pdf_sum)

# individual curves
pdf_individual = probabilities * pdf_sum.reshape(-1, 1)
plt.plot(x_vals, pdf_individual)

plt.xlabel('marker distance (mm)')
plt.ylabel('frequency')
plt.title('real distribution, total model, and model components')
plt.legend(['total model (= sum)','distribution component 1','distribution component 2'])
plt.show()


# test non-normal distribution with n=10




# visualize cluster in time plot
gmm = GaussianMixture(n_components=2, covariance_type="full", tol=0.001)
gmm.fit(X=dist_1.reshape(-1,1))

clust_label = gmm.predict(dist_1.reshape(-1, 1))

plt.figure()
dist_1 = np.linalg.norm(ac[0]-ac[1],axis=0)
times = np.arange(0, csvdata.shape[1]/120, 1/120)
plt.scatter(times, dist_1, c=clust_label, s=2)
plt.xlabel('measuring time (s)')
plt.ylabel('distance (mm)')
plt.title('marker 1 - marker 2')
plt.show()




##################  better idea: use 2D information  ##########################

dist_1 = np.linalg.norm(ac[0]-ac[1],axis=0)
dist_2 = np.linalg.norm(ac[0]-ac[2],axis=0)

pd.DataFrame(dist_1).hist(bins=100, density=True)
plt.xlabel('marker distance (mm)')
plt.ylabel('number of data points')
plt.title('marker 1 - marker 2')

pd.DataFrame(dist_2).hist(bins=100, density=True)
plt.xlabel('marker distance (mm)')
plt.ylabel('number of data points')
plt.title('marker 1 - marker 3')


dist_2D = np.stack((dist_1, dist_2)).T

gmm2D = GaussianMixture(n_components=2, covariance_type="full", tol=0.001)
gmm2D.fit(X=dist_2D)

# distance over distance
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
clust_label2D = gmm2D.predict(dist_2D)
plt.scatter(dist_2D[:, 0], dist_2D[:, 1], c=clust_label2D, s=5, cmap='viridis');
plt.xlabel('distance 1 (mm)')
plt.ylabel('distance 2 (mm)')
plt.title('markers 1, 2, and 3')
plt.show()


# plot over time
plt.figure()
dist_1 = np.linalg.norm(ac[0]-ac[1],axis=0)
plt.scatter(times, dist_1, c=clust_label2D, s=5)
plt.xlabel('measuring time (s)')
plt.ylabel('distance (mm)')
plt.title('marker 1 - marker 2')
plt.show()








###############################  3D example  ##################################
#           (not much different from 2D for this dataset)
        
        
dist_1 = np.linalg.norm(ac[0]-ac[1],axis=0)
dist_2 = np.linalg.norm(ac[0]-ac[2],axis=0)
dist_3 = np.linalg.norm(ac[1]-ac[2],axis=0)
dist_3D = np.stack((dist_1, dist_2, dist_3)).T



gmm3D = GaussianMixture(n_components=2, covariance_type="full", tol=0.001)
gmm3D.fit(X=dist_3D)
clust_label3D = gmm3D.predict(dist_3D)


from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(dist_1, dist_2, dist_3, c=clust_label3D, s=5)
ax.set_xlabel('distances 1')
ax.set_ylabel('distances 2')
ax.set_zlabel('distances 3')
plt.show()



#############  sensible number of components / covar. type  ###################

# using information criterion (aic or bic) gmm.aic(X) or gmm.bic(X)
# adopted from: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html



lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:
    for n_components in n_components_range:
        
        gmm = GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(dist_2D)
        bic.append(gmm.bic(dist_2D))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
            
bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []
           
            
            
# Plot the BIC scores
plt.figure()
ax = plt.subplot()
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per covariance-type and number of components')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
ax.set_xlabel('Number of components')
ax.legend([b[0] for b in bars], cv_types)

