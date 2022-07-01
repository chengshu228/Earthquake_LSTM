import config

math = config.math
np = config.np
pd = config.pd
datetime = config.datetime
relativedelta = config.relativedelta
LinearRegression = config.LinearRegression

min_year = config.min_year
max_year = config.max_year
min_latitude = config.min_latitude
min_longitude = config.min_longitude
min_magnitude = config.min_magnitude
min_number = config.min_number
span_lat = config.span_lat
span_lon = config.span_lon
time_window = config.time_window
next_month = config.next_month
blocks = config.blocks
features = config.features
index = config.index
energy = config.energy
n_splits = config.n_splits
layer = config.layer
layer_size = config.layer_size
rate = config.rate
weight = config.weight
epochs = config.epochs
learning_rate = config.learning_rate
filename = config.filename
catolog_name = config.catolog_name
each_move = config.each_move
file_location = config.file_location

time = config.time
start_time = time.time()

data = pd.read_csv(file_location+r'\catalog'+r'\filter_catalog.txt', 
    dtype=str, delimiter=' ', header=None)
data = np.array(data)   
date = pd.DataFrame({'year': data[:,0], 'month': data[:,1], 'day': data[:,2], 
    'hour': data[:,3], 'minute': data[:,4], 'second': data[:,5]})
date = pd.to_datetime(date)
latitude = np.around(np.array(data[:,-3], dtype=np.float32), 3)
longitude = np.around(np.array(data[:,-2], dtype=np.float32), 3)
magnitude = np.around(np.array(data[:,-1], dtype=np.float32), 1)
min_lat, mid_lat = min_latitude, max(latitude)-span_lat
min_lon, mid_lon = min_longitude, max(longitude)-span_lon
E = np.around(10**(11.8+magnitude*1.5), 0)

with open(file_location+r'\factor'+r'\factor-'+catolog_name+r'.txt', mode='w+', encoding='utf-8') as fout:
    for y in np.arange(min_year, max_year+1):
        print('year:', y)
        for m in np.arange(1, 12+1):
            for lat in np.arange(mid_lat, min_lat-each_move, -each_move):
                for lon in np.arange(min_lon, mid_lon+each_move, each_move):
                    if datetime.datetime(y,m,1,0,0,0) + relativedelta(months=time_window+next_month) \
                         >= datetime.datetime(2021,3,1,0,0,0):  
                        break
                    else:
                        date_begin = datetime.datetime(y,m,1,0,0,0)
                        date_end = datetime.datetime(y,m,1,0,0,0) + relativedelta(months=time_window)
                        flag_year = np.logical_and(date>date_begin, date<=date_end)
                        flag_lat = np.logical_and(latitude>=lat, latitude<=lat+span_lat)
                        flag_lon = np.logical_and(longitude>=lon, longitude<=lon+span_lon)
                        flag_location = np.logical_and(flag_lat, flag_lon)
                        flag = np.logical_and(flag_year, flag_location)
                        mag = magnitude[flag]

                        date_end_next = datetime.datetime(y,m,1,0,0,0) + relativedelta(months=time_window+next_month)
                        flag_year_next = np.logical_and(date>date_end, date<=date_end_next)
                        flag_lat_next = np.logical_and(latitude>=lat, latitude<=lat+span_lat)
                        flag_lon_next = np.logical_and(longitude>=lon, longitude<=lon+span_lon)
                        flag_location_next = np.logical_and(flag_lat_next, flag_lon_next)
                        flag_next = np.logical_and(flag_year_next, flag_location_next)
                        mag_next = magnitude[flag_next]

                        if not (mag_next.size>0 and mag.size>0 and len(mag)>=min_number):
                            print('Warning: ', 'next: ', mag_next.size>0, '>=min_number: ', len(mag)>=min_number)
                        else:
                            max_magnitude_next = np.max(mag_next)
                            max_magnitude = np.max(mag)
                            mean_magnitude = np.mean(mag)  
                            frequency = len(mag)   

                            m_lstsq = np.arange(min_magnitude, max_magnitude+0.05, 0.1)
                            n_lstsq = np.zeros_like(m_lstsq)
                            
                            for i,element in enumerate(m_lstsq):
                                n_lstsq[i] = np.sum(mag>=element)

                            if n_lstsq.any() == 0:
                                print('n_lstsq=', n_lstsq)
                        
                            # b值最小二乘法
                            b_lstsq = len(n_lstsq)*np.sum(m_lstsq*np.log10(n_lstsq)) - \
                                np.sum(m_lstsq)*np.sum(np.log10(n_lstsq))
                            b_lstsq /= (np.sum(m_lstsq)**2 - len(n_lstsq)*np.sum(m_lstsq**2))

                            # a值最小二乘法
                            a_lstsq = np.sum(np.log10(n_lstsq)+b_lstsq*m_lstsq) / len(n_lstsq)

                            # b值最大似然估计法
                            b_mle = (np.log10(math.exp(1)) / (mean_magnitude-min_magnitude))

                            # if b_lstsq>1.7 or b_lstsq<0.52:
                            #     import matplotlib.pyplot as plt
                            #     fig = plt.figure(figsize=(7, 5))
                            #     plt.subplot(1,1,1) 
                            #     plt.scatter(m_lstsq, np.log10(n_lstsq), c='', edgecolor='dodgerblue')
                            #     plt.plot(m_lstsq, a_lstsq-b_lstsq*m_lstsq, 'b',linewidth=2)
                            #     plt.text(1.8, 0.5, 'logN={:.4f}-{:.4f}*M'.format(a_lstsq, b_lstsq), fontsize=14)
                            #     plt.xlabel('M', fontsize=14)
                            #     plt.ylabel('logN', fontsize=14)
                            #     plt.title('logN=$a_{lstsq}$-$b_{lstsq}$*M', fontsize=14)
                            #     plt.xticks(size=12)
                            #     plt.yticks(size=12)
                            #     plt.grid(True, linestyle='--', linewidth=1.5)
                            #     ax = plt.gca()
                            #     ax.spines['bottom'].set_linewidth(1.5)
                            #     ax.spines['left'].set_linewidth(1.5)
                            #     ax.spines['top'].set_linewidth(1.5)
                            #     ax.spines['right'].set_linewidth(1.5)
                            #     plt.show()

                            # 最大震级欠缺
                            max_mag_absence = max_magnitude - (a_lstsq/b_lstsq)

                            # 最小二乘法G-R方程拟合时的均方根误差
                            rmse_lstsq = np.sqrt(np.sum(np.power(np.log10(n_lstsq)-\
                                (a_lstsq-b_lstsq*m_lstsq),2)) / len(n_lstsq)) 

                            # 平均纬度
                            mean_lat = np.mean(latitude[flag])
                            # 与平均纬度的均方差
                            rmse_lat = np.sqrt(np.sum(np.power(latitude[flag]-mean_lat, 2)) / frequency)

                            # 平均经度
                            mean_lon = np.mean(longitude[flag])
                            # 与平均经度的均方差
                            rmse_lon = np.sqrt(np.sum(np.power(longitude[flag]-mean_lon, 2)) / frequency)

                            # 斜率
                            model = LinearRegression()
                            model.fit(np.array(longitude[flag]).reshape(-1,1), 
                                np.array(latitude[flag]).reshape(-1,1))
                            k = model.coef_
                            k = round(float(k), 4)

                            # 能量平方根
                            energy_square = np.sqrt(E[flag])
                            total_energy_square = np.sum(energy_square)

                            # 能量加权的震中平均纬度
                            epicenter_latitude = np.sum(latitude[flag]*energy_square) / total_energy_square
                            
                            # 能量加权的震中平均经度
                            epicenter_longitude = np.sum(longitude[flag]*energy_square) / total_energy_square                  

                            fout.write('{0:.1f} {1:.0f} {2:.1f} {3:.2f} {4:.4f} {5:.4f} {6:.4f} {7:.4f} {8:.4f} {9:.2f} {10:.4f} {11:.4f} {12:.4f} {13:.4f} {14:.4f} {15:.4f} {16:.4f}\n'.\
                                format(max_magnitude_next, frequency, max_magnitude, mean_magnitude, \
                                b_lstsq, b_mle, a_lstsq, \
                                max_mag_absence, rmse_lstsq, total_energy_square, \
                                mean_lon, rmse_lon, mean_lat, rmse_lat, \
                                k, epicenter_longitude, epicenter_latitude))  

print((time.time()-start_time)/60, 'minutes')


# E = np.zeros_like(magnitude)
# for i in np.arange(len(data)):
#     if data[i,-2]=='ML':  # E=10**(1.8*ML+12)
#         E[i] = np.around(np.power(10, 1.8*float(data[i,-1])+12), 2)
#     elif data[i,-2]=='mb':  # E=10**(2.4*mb+5.8)
#         E[i] = np.around(np.power(10, 2.4*float(data[i,-1])+5.8), 2)
#     elif data[i,-2]=='Ms':  # E=10**(1.5*Ms+11.4)
#         E[i] = np.around(np.power(10, 1.5*float(data[i,-1])+11.4), 2)
