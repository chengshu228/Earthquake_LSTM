print('\tBegin merge_catalog.py')

import config
import numpy as np

catalog_location = config.catalog_location
data_location = config.data_location
min_year = config.min_year
max_year = config.max_year 

catalog_merge = np.empty(shape=(0, 9))
for year in np.arange(min_year, max_year+1, 1):
    print(year)
    year = str(year)
    catalog = globals()
    catalog[year] = np.loadtxt(
        catalog_location+f'\\{year}.txt', dtype=str)
    catalog[year] = np.delete(catalog[year], \
        [6,7,9,12,13,14,15,16], axis=1).reshape(-1, 9).astype(np.float32)
    if year==1932: catalog_merge = catalog[year]
    else:
        catalog_merge = np.concatenate(
            (catalog_merge, catalog[year]), axis=0)

catalog_merge[:,0] = np.around(catalog_merge[:,0],0)
catalog_merge[:,1] = np.around(catalog_merge[:,1],0)
catalog_merge[:,2] = np.around(catalog_merge[:,2],0)
catalog_merge[:,3] = np.around(catalog_merge[:,3],0)
catalog_merge[:,4] = np.around(catalog_merge[:,4],0)
catalog_merge[:,5] = np.around(catalog_merge[:,5],2)
catalog_merge[:,6] = np.around(catalog_merge[:,6],1)
catalog_merge[:,7] = np.around(catalog_merge[:,7],3)
catalog_merge[:,8] = -np.around(catalog_merge[:,8],3)

np.savetxt(data_location+r'\merge_catalog.txt', 
    catalog_merge, fmt='%.3f', delimiter=' ')

print('\tEnd merge_catalog.py\n')

