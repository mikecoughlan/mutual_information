import os
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import seaborn as sns

import scipy.stats as stats
import matplotlib.pyplot as plt



def loading_solarwind(source='ace', limit_to_twins=False):
	'''
	Loads the solar wind data

	Returns:
		df (pd.dataframe): dataframe containing the solar wind data
	'''

	print('Loading solar wind data....')
	if source == 'omni':
		df = pd.read_feather('../data/SW/omniData.feather')
		df.set_index('Epoch', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
	elif source=='ace':
		df = pd.read_feather('../data/SW/ace_data.feather')
		df.set_index('ACEepoch', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
	else:
		raise ValueError('Invalid source')
	if limit_to_twins:
		df = df[pd.to_datetime('2009-07-20'):pd.to_datetime('2017-12-31')]

	return df


def loading_supermag(station):
	'''
	Loads the supermag data

	Args:
		station (string): station of interest

	Returns:
		df (pd.dataframe): dataframe containing the supermag data with a datetime index
	'''

	print(f'Loading station {station}....')
	df = pd.read_feather(supermag_dir+station+'.feather')

	# limiting the analysis to the nightside
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
	df['theta'] = (np.arctan2(df['N'], df['E']) * 180 / np.pi)	# calculates the angle of B_H
	df['cos_theta'] = np.cos(df['theta'] * np.pi / 180)			# calculates the cosine of the angle of B_H
	df['sin_theta'] = np.sin(df['theta'] * np.pi / 180)			# calculates the sine of the angle of B_H

	return df

class RegionPreprocessing():

	def __init__(self, cluster=None, region=None, features=None, mean=False, std=False, maximum=False, median=False, limit_to_twins=False, **kwargs):

		if cluster is None:
			raise ValueError('Must specify a cluster to analyze.')

		if region is None:
			raise ValueError('Must specify a region to analyze.')

		self.cluster = cluster
		self.region_name = region
		self.features = features
		self.mean = mean
		self.std = std
		self.maximum = maximum
		self.median = median
		self.limit_to_twins = limit_to_twins

		self.__dict__.update(kwargs)


	def loading_supermag(self, station):
		'''
		Loads the supermag data

		Args:
			station (string): station of interest

		Returns:
			df (pd.dataframe): dataframe containing the supermag data with a datetime index
		'''

		print(f'Loading station {station}....')
		df = pd.read_feather(supermag_dir+station+'.feather')

		# limiting the analysis to the nightside
		df.set_index('Date_UTC', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
		df['theta'] = (np.arctan2(df['N'], df['E']) * 180 / np.pi)	# calculates the angle of B_H
		df['cos_theta'] = np.cos(df['theta'] * np.pi / 180)			# calculates the cosine of the angle of B_H
		df['sin_theta'] = np.sin(df['theta'] * np.pi / 180)			# calculates the sine of the angle of B_H

		return df



	def getting_dbdt_dataframe(self):

		if self.limit_to_twins:
			dbdt_df = pd.DataFrame(index=pd.date_range(start='2009-07-20', end='2017-12-31 23:59:00', freq='min'))
		else:
			dbdt_df = pd.DataFrame(index=pd.date_range(start='1995-01-01', end='2019-12-31 23:59:00', freq='min'))

		for station in self.region['stations']:
			# loading the station data
			station_df = pd.read_feather(supermag_dir + station + '.feather')
			station_df.set_index('Date_UTC', inplace=True)
			station_df.index = pd.to_datetime(station_df.index)
			# creating the dbdt time series
			dbdt_df[station] = station_df['dbht']

		return dbdt_df


	def finding_mlt(self):
		'''finding which station has the least missing data and using that to define the mlt for the region'''

		print(f'region keys: {self.region.keys()}')
		if 'mlt_station' in self.region.keys():
			print(f'MLT station already defined for region {self.region_name}')
			return self.mlt_df[self.clusters[self.cluster]['regions'][self.region_name]['mlt_station']]

		else:
			temp_df = self.mlt_df.copy()

			storm_list = pd.read_feather('outputs/regular_twins_map_dates.feather', columns=['dates'])
			storm_list = storm_list['dates']

			stime, etime, storms = [], [], []					# will store the resulting time stamps here then append them to the storm time df

			# will loop through the storm dates, create a datetime object for the lead and recovery time stamps and append those to different lists
			for date in storm_list:
				if isinstance(date, str):
					date = pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S')
				stime.append(date.round('T')-pd.Timedelta(minutes=30))
				etime.append(date.round('T')+pd.Timedelta(minutes=9))
				# else:
				# 	stime.append(date.round('T')-pd.Timedelta(minutes=30))
				# 	etime.append(date.round('T')+pd.Timedelta(minutes=9))

			for start, end in zip(stime, etime):		# looping through the storms to remove the data from the larger df
				if start < temp_df.index[0] or end > temp_df.index[-1]:						# if the storm is outside the range of the data, skip it
					continue
				storm = temp_df[(temp_df.index >= start) & (temp_df.index <= end)]

				if len(storm) != 0:
					storms.append(storm)

			all_storms = pd.concat(storms, axis=0)
			storm.reset_index(drop=True, inplace=True)		# resetting the storm index and simultaniously dropping the date so it doesn't get trained on


			# self.mlt_df['mix'] = self.mlt_df.median(axis=1)
			missing_mlt = temp_df.isnull().sum()
			station = missing_mlt.idxmin()

			print(f'Missing data for each station: {missing_mlt}')
			print(f'Station with the least missing data: {station}')

			self.clusters[self.cluster]['regions'][self.region_name]['mlt_station'] = station

			return self.mlt_df[station]


	def calculating_rsd(self):

		dbdt_df = self.getting_dbdt_dataframe()
		rsd = pd.DataFrame(index=dbdt_df.index)

		# calculating the RSD
		for col in dbdt_df.columns:
			ss = dbdt_df[col]
			temp_df = dbdt_df.drop(col,axis=1)
			ra = temp_df.mean(axis=1)
			rsd[col] = ss-ra

		max_rsd = rsd.max(axis=1)
		max_station = rsd.idxmax(axis=1)
		rsd['max_rsd'] = max_rsd
		rsd['max_station'] = max_station

		return rsd


	def combining_stations_into_regions(self, map_keys=None):

		if self.limit_to_twins:
			start_time = pd.to_datetime('2009-07-20')
			end_time = pd.to_datetime('2017-12-31')
		else:
			start_time = pd.to_datetime('1995-01-01')
			end_time = pd.to_datetime('2019-12-31')
		time_period = pd.date_range(start=start_time, end=end_time, freq='min')

		regional_df = pd.DataFrame(index=time_period)
		self.mlt_df = pd.DataFrame(index=time_period)
		self.lons_dict = {}

		# creating a dataframe for each feature with the twins time period as the index and storing them in a dict
		feature_dfs = {}
		if self.features is not None:
			for feature in self.features:
				feature_dfs[feature] = pd.DataFrame(index=time_period)

		for stat in self.region['stations']:
			df = self.loading_supermag(stat)
			self.lons_dict[stat] = df['GEOLON'].loc[df['GEOLON'].first_valid_index()]
			df = df[start_time:end_time]
			self.mlt_df[stat] = df['MLT']
			if self.features is not None:
				for feature in self.features:
					feature_dfs[feature][f'{stat}_{feature}'] = df[feature]
		if self.features is not None:
			for feature in self.features:
				if self.mean:
					if feature == 'N' or feature == 'E':
						regional_df[f'{feature}_mean'] = feature_dfs[feature].abs().mean(axis=1)
					else:
						regional_df[f'{feature}_mean'] = feature_dfs[feature].mean(axis=1)
				if self.std:
					regional_df[f'{feature}_std'] = feature_dfs[feature].std(axis=1)
				if self.maximum:
					if feature == 'N' or feature == 'E':
						regional_df[f'{feature}_max'] = feature_dfs[feature].abs().max(axis=1)
					else:
						regional_df[f'{feature}_max'] = feature_dfs[feature].max(axis=1)
				if self.median:
					if feature == 'N' or feature == 'E':
						regional_df[f'{feature}_median'] = feature_dfs[feature].abs().median(axis=1)
					else:
						regional_df[f'{feature}_median'] = feature_dfs[feature].median(axis=1)

		mlt = self.finding_mlt()
		rsd = self.calculating_rsd()

		regional_df['rsd'] = rsd['max_rsd']
		regional_df['MLT'] = mlt
		regional_df['cosMLT'] = np.cos(regional_df['MLT'] * 2 * np.pi * 15 / 360)
		regional_df['sinMLT'] = np.sin(regional_df['MLT'] * 2 * np.pi * 15 / 360)


		if map_keys is not None:
			regional_df = regional_df[regional_df.index.isin(map_keys)]

		return regional_df


	def __call__(self, cluster_dict='cluster_dict.pkl', **kwargs):

		with open(cluster_dict, 'rb') as f:
			self.clusters = pickle.load(f)

		self.region = self.clusters[self.cluster]['regions'][self.region_name]

		regional_df = self.combining_stations_into_regions()

		with open(cluster_dict, 'wb') as f:
			pickle.dump(self.clusters, f)

		return regional_df