
import pandas as pd
import numpy as np

file_location = r'C:\Users\cshu\Desktop\shi\Japan'

catalog = pd.read_csv(file_location+r'\Japan\h1919.txt', 
    dtype=str, delimiter='#', header=None)
catalog = np.array(catalog)
print(catalog.shape)

year = np.array(catalog[:,1], dtype=int)*1000 + \
    np.array(catalog[:,2], dtype=int)*100 + \
    np.array(catalog[:,3], dtype=int)*10 + \
    np.array(catalog[:,4], dtype=int)
print('year:', min(year), max(year))

month = np.array(catalog[:,5], dtype=int)*10 + \
    np.array(catalog[:,6], dtype=int)
print('month:', min(month), max(month))

day = np.array(catalog[:,7], dtype=int)*10 + \
    np.array(catalog[:,8], dtype=int)
print('day:', min(day), max(day))

hour = np.array(catalog[:,9], dtype=int)*10 + \
    np.array(catalog[:,10], dtype=int)
print('hour:', min(hour), max(hour))

minute = np.array(catalog[:,11], dtype=int)*10 + \
    np.array(catalog[:,12], dtype=int)
print('minute:', min(minute), max(minute))

catalog[:,13][np.argwhere(catalog[:,13]==' ')] = 0
catalog[:,14][np.argwhere(catalog[:,14]==' ')] = 0
catalog[:,15][np.argwhere(catalog[:,15]==' ')] = 0
catalog[:,16][np.argwhere(catalog[:,16]==' ')] = 0
sec = np.array(catalog[:,13], dtype=int)*10 +\
    np.array(catalog[:,14], dtype=int) + \
    np.array(catalog[:,15], dtype=int)*0.1 + \
    np.array(catalog[:,16], dtype=int)*0.01
print('sec:', min(sec), max(sec))

lat = []
for i in np.arange(len(catalog)):
    if catalog[i, 21] == '-':
        if catalog[i, 22] == '0' or catalog[i, 22] == ' ':
            lat_temp = float(catalog[i, 23])*(-1)
            lat = np.append(lat, lat_temp)
        elif catalog[i, 22] == '-':
            lat_temp = float(catalog[i, 23])*(-1)
            lat = np.append(lat, lat_temp)
        else:
            lat_temp = (int(catalog[i, 22])*10 + int(catalog[i, 23]))*(-1)
            lat = np.append(lat, lat_temp)
    else:
        if catalog[i, 22] == '0' or catalog[i, 22] == ' ':
            lat_temp = float(catalog[i, 23])
            lat = np.append(lat, lat_temp)
        elif catalog[i, 22] == '-':
            lat_temp = float(catalog[i, 23])*(-1)
            lat = np.append(lat, lat_temp)
        else:
            lat_temp = int(catalog[i, 22])*10 + int(catalog[i, 23])
            lat = np.append(lat, lat_temp)
print('lat:', min(lat), max(lat))

lon = []
for i in np.arange(len(catalog)):
    if catalog[i, 32] == '-':
        if catalog[i, 33] == '0' or catalog[i, 33] == ' ':
            lon_temp = (float(catalog[i, 34])*100 + float(catalog[i, 35])*10 + \
                float(catalog[i, 36]))*(-1)
            lon = np.append(lon, lon_temp)
        elif catalog[i, 33] == '-':
            if catalog[i, 34] == '0' or catalog[i, 34] == ' ' or catalog[i, 34] == '-':
                lon_temp = (float(catalog[i, 35])*10 + \
                    float(catalog[i, 36]))*(-1)
                lon = np.append(lon, lon_temp)
             else:
                lon_temp = float(catalog[i, 35])*10 + \
                    float(catalog[i, 36])
                lon = np.append(lon, lon_temp)
        else:
            lon_temp = (float(catalog[i, 33])*100 + float(catalog[i, 34])*100 + \
                float(catalog[i, 35])*10 + float(catalog[i, 36]))*(-1)
            lon = np.append(lon, lon_temp)
    else:
        if catalog[i, 22] == '0' or catalog[i, 22] == ' ':
            lon_temp = float(catalog[i, 23])
            lon = np.append(lon, lon_temp)
        elif catalog[i, 22] == '-':
            lon_temp = float(catalog[i, 23])*(-1)
            lon = np.append(lon, lon_temp)
        else:
            lon_temp = int(catalog[i, 22])*10 + int(catalog[i, 23])
            lon = np.append(lon, lon_temp)
print('lon:', min(lon), max(lon))

# lat1 = np.array(catalog[:,24], dtype=int)*1000 +\
#     np.array(catalog[:,25], dtype=int)*100 + \
#     np.array(catalog[:,26], dtype=int)*10 +\
#     np.array(catalog[:,27], dtype=int)
# print('lat1:', min(lat1), max(lat1))

# catalog[:,33][np.argwhere(catalog[:,32:33+1]=='- ')] = '0'

# catalog[:,32:33+1][np.argwhere(catalog[:,32:33+1]=='- ')] = '-0'
# catalog[:,33:34+1][np.argwhere(catalog[:,32:34+1]=='-  ')] = '-00'
# lon = np.array(catalog[:,32], dtype=int)*1000 +\
#     np.array(catalog[:,33], dtype=int)*100 + \
#     np.array(catalog[:,34], dtype=int)*10 +\
#     np.array(catalog[:,35], dtype=int)
# print('lon:', min(lon), max(lon))

# lon1 = np.array(catalog[:,36], dtype=int)*1000 +\
#     np.array(catalog[:,37], dtype=int)*100 + \
#     np.array(catalog[:,38], dtype=int)*10 +\
#     np.array(catalog[:,39], dtype=int)
# print('lon1:', min(lon1), max(lon1))





