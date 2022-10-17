print('Begin merge_catalog.py')

import numpy as np
import config

catalog_location,data_location = config.catalog_location,config.data_location
min_year,max_year = config.min_year,config.max_year 

catalog_merge = np.empty(shape=(0, 9))
for year in np.arange(min_year, max_year+1, 1):
    print(year)
    year = str(year)
    catalog = globals()
    catalog[year] = np.genfromtxt(
        catalog_location+f'\\{year}.catalog'.format(year), dtype=str)
    
    print(catalog[year])
    catalog[year] = np.delete(catalog[year], \
        [6,7,9,12,13,14,15,16], axis=1).reshape(-1,9).astype(np.float32)
    if year==1932: catalog_merge = catalog[year]
    else:
        catalog_merge = np.concatenate(
            (catalog_merge,catalog[year]), axis=0)

la = [0,0,0,0,0,2,1,3,3]
for i in np.arange(catalog_merge.shape[1]):
    if i==catalog_merge.shape[1]-1:
        catalog_merge[:,i] = -np.around(catalog_merge[:,i],la[i])
    else:
        catalog_merge[:,i] = np.around(catalog_merge[:,i],la[i])

np.savetxt(data_location+f'\\merge_catalog_{min_year}_{max_year}.txt', 
    catalog_merge, fmt='%.3f', delimiter=' ')

print('End merge_catalog.py')

