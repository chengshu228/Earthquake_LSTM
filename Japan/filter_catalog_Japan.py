print('\n\tBegin filter_catalog_Japan.py')

import config

np = config.np
pd = config.pd
tf = config.tf
min_year = config.min_year
max_year = config.max_year 
min_latitude = config.min_latitude 
max_latitude = config.max_latitude 
min_longitude = config.min_longitude
max_longitude = config.max_longitude
min_magnitude = config.min_magnitude 

time = config.time
start_time = time.time()

file_location = config.file_location

catalog = pd.read_csv(file_location+r'\catalog\JMA-alltype-cata.dat', 
    delimiter=' ', header=None)
catalog = np.array(catalog)
print(catalog.shape)

catalog = np.concatenate((
    catalog[:, 0:4+1].reshape(-1, 5), 
    np.around(catalog[:, 5]/100, 2).reshape(-1, 1),
    np.around(catalog[:, 6] + catalog[:, 7]/6000, 4).reshape(-1, 1),
    np.around(catalog[:, 8] + catalog[:, 9]/6000, 4).reshape(-1, 1),
    (catalog[:, -2]/10).reshape(-1, 1),
    ), axis=1)

print('\tcatalog\n\t原始地震目录({}-{}):'.format(min_year, max_year), catalog.shape)

print('\n\tyear:', min(catalog[:, 0]), max(catalog[:, 0]),
    '\n\tmonth:', min(catalog[:, 1]), max(catalog[:, 1]),
    '\n\tday:', min(catalog[:, 2]), max(catalog[:, 2]),
    '\n\thour:', min(catalog[:, 3]), max(catalog[:, 3]),
    '\n\tminute:', min(catalog[:, 4]), max(catalog[:, 4]),
    '\n\tsecond', min(catalog[:, 5]), max(catalog[:, 5]),
    '\n\tlatitude:', min(catalog[:, 6]), max(catalog[:, 6]),
    '\n\tlongitude:', min(catalog[:, 7]), max(catalog[:, 7]),
    '\n\tMagnitude:', min(catalog[:, 8]), max(catalog[:, 8]),)

import matplotlib.pyplot as plt
fonten = config.fonten
fig = plt.figure(figsize=(14, 6))   
plt.title(r'latitude - longitude', fontdict=fonten, fontsize=20, color='red')
fig.add_subplot(1,1,1)
plt.grid(True, linestyle='--', linewidth=1)
plt.scatter(catalog[:,-2], catalog[:, -3], c='', edgecolor='dodgerblue')
plt.xlabel('longitude', fontproperties='Arial', fontsize=18)  
plt.ylabel('latitude', fontdict=fonten, fontsize=18)  
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)   
plt.tight_layout()
# plt.savefig(r'.\figure\{}-Histgram-test.png'.format(filename))
plt.show()

catalog = np.delete(catalog, list(np.argwhere(catalog[:, 0]<min_year)) 
    + list(np.argwhere(catalog[:, 0]>max_year))
    + list(np.argwhere(catalog[:, -3]>max_latitude)) 
    + list(np.argwhere(catalog[:, -3]<min_latitude))
    + list(np.argwhere(catalog[:, -2]>max_longitude)) 
    + list(np.argwhere(catalog[:, -2]<min_longitude))
    + list(np.argwhere(catalog[:, -1]<min_magnitude)), 
    axis=0).reshape(-1, catalog.shape[1])

print('\t处理后的地震目录:', catalog.shape, 
    '\n\tyear:', min(catalog[:, 0]), max(catalog[:, 0]),
    '\n\tmonth:', min(catalog[:, 1]), max(catalog[:, 1]),
    '\n\tday:', min(catalog[:, 2]), max(catalog[:, 2]),
    '\n\thour:', min(catalog[:, 3]), max(catalog[:, 3]),
    '\n\tminute:', min(catalog[:, 4]), max(catalog[:, 4]),
    '\n\tsecond', min(catalog[:, 5]), max(catalog[:, 5]),
    '\n\tlatitude:', min(catalog[:, 6]), max(catalog[:, 6]),
    '\n\tlongitude:', min(catalog[:, 7]), max(catalog[:, 7]),
    '\n\tMagnitude:', min(catalog[:, 8]), max(catalog[:, 8]),)

np.savetxt(file_location+r'\catalog\filter_catalog.txt', 
    catalog, fmt='%f', delimiter=' ')

print('\n\tEnd filter_catalog_Japan.py')
print('\t', (time.time()-start_time)/60, 'minutes')


# # 将地震目录中的面波震级转化为里氏震级,并统一震级 
# # ML=(MS+1.08)/1.13  ML=(1.17MB+0.67)/1.13
# for i in np.arange(len(data)):
#     # if data[i, -2]=='ML':
#     #     data[i, -1] = np.around((data[i, -1]+1.08)/1.13, 1)
#     # elif data[i, -2]=='mb':
#     #     data[i, -1] = np.around((1.17*data[i, -1]+0.67)/1.13, 1)
#     # else:
#     #     data[i, -1] = data[i, -1]

