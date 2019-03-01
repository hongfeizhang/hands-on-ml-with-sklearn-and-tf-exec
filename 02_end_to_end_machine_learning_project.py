import sys
import os
import tarfile

HOUSING_PATH	=	"datasets/housing"

import	pandas	as	pd
def	load_housing_data(housing_path=HOUSING_PATH):
    csv_path	=	os.path.join(housing_path,	"housing.csv")
    return	pd.read_csv(csv_path)

housing=load_housing_data()
print(housing.head())

import	matplotlib.pyplot	as	plt
housing.hist(bins=50,	figsize=(20,15))
plt.show()

