import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

import knncmi
import TE
import utils

# Load data
RP = utils.RegionPreprocessing(cluster='greenland_cluster', region='GRL-0')
data = RP()
data = data.dropna()

sw_df = utils.loading_solarwind()

def make_some_noise(X,Y,Z=None, t_entropy=False, delay=None, n_samples=100, random_state=42):
	random_MIs = []
	np.random.seed(random_state)
	for i in range(n_samples):
		# shuffling the X array
		X_rand = np.random.permutation(X)
		if Z is not None:
			mi_val = utils.conditional_mutual_inforamtion(X_rand, Y, Z)
		else:
			if t_entropy:
				mi_val = TE.transfer_entropy(Y, X_rand, delay=delay)
			else:
				mi_val = utils.mutual_information(X_rand, Y)
		random_MIs.append(mi_val)
	return np.mean(random_MIs), np.std(random_MIs)


mis, rand_means, rand_stds = [], [], []
for i in range(1, 121, 5):
	stime = time.time()
	temp_data = pd.concat([data, sw_df], axis=1)
	# temp_data['shifted_rsd'] = data['rsd'].shift(-i)
	temp_data.dropna(inplace=True)

	x = 'Vx'
	y = 'rsd'
	z = 'B_Total'

# data = data[[x,y,z]]
	mu, sigma = make_some_noise(temp_data[x].to_numpy(), temp_data[y].to_numpy(), t_entropy=True, delay=i, n_samples=100)
	rand_means.append(mu)
	rand_stds.append(sigma)
	mi = TE.transfer_entropy(temp_data[y].to_numpy(), temp_data[x].to_numpy(), delay=i)
	# mi = utils.conditional_mutual_inforamtion(temp_data[x].to_numpy(), temp_data[y].to_numpy(), temp_data[z].to_numpy())
	etime = time.time()

	print(f'MI manual calculation: {mi} - time: {round(etime-stime, 2)}s')
	mis.append(mi)
	# cmi = utils.conditional_mutual_inforamtion(tempdata[x].to_numpy(), tempdata[y].to_numpy(), tempdata[z].to_numpy())
	# print(f'CMI manual calculation: {cmi}')
# mi2 = mutual_info_regression(data[[x, y]], data[y])
# print(f'MI sklearn calculation 1: {mi2}')

	# n_bins = int(math.log2(len(temp_data)) + 1)

	# # X = np.histogram(temp_data[x], bins=n_bins)[0]
	# # Y = np.histogram(temp_data[y], bins=n_bins)[0]
	# X = temp_data[x].to_numpy()
	# Y = temp_data[y].to_numpy()
	# # Z = np.histogram(data[z], bins=n_bins)[0]
	# # inputs = np.array([X, Z])
	# MI2 = mutual_info_regression(X.reshape(-1,1), Y, n_neighbors=3, random_state=42)
	# print(f'MI sklearn calculation 2: {MI2}')
	# mis.append(MI2[0])
# print(f'MI manual calculation: {MI2[1]}')

plt.plot(range(0,120, 5), mis)
plt.errorbar(range(0,120, 5), rand_means, yerr=rand_stds)
plt.show()
raise ValueError

# MI = mutual_info_regression(data[[x, z]], data[y])


print(len(data))
print(n_bins)

X = np.histogram(data[x], bins=n_bins)[0]
Y = np.histogram(data[y], bins=n_bins)[0]
Z = np.histogram(data[z], bins=n_bins)[0]
inputs = np.array([X, Y])
MI2 = mutual_info_regression(inputs.T, Z)
print(MI)
print(MI2)

data = data[[x, y, z]].to_numpy()
# getting cmi
value = knncmi.cmi([0], [1], [2], k=3, data=data, discrete_dist=1, minzero=1)
print(value)
