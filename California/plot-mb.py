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
magnitude = np.around(np.array(data[:,-3], dtype=np.float32), 1)

Mc = np.arange(-0.01, 6.01, 0.1)
# Mc = np.arange(-0.01, 7.21, 0.1)
b = []
for Mtry in Mc:
    with open(file_location+r'\factor'+r'\factor-'+catolog_name+r'.txt', \
        mode='w+', encoding='utf-8') as fout:

        min_magnitude = Mtry
        
        magnitude0 = magnitude[np.where(magnitude>=Mtry-0.05)]
        mean_magnitude = np.mean(magnitude0)  

        # b值最大似然估计法
        b_mle = (np.log10(math.exp(1)) / (mean_magnitude-min_magnitude))
        b.append(b_mle)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 5))
plt.subplot(1,1,1) 
plt.plot(Mc, b, 'dodgerblue', linewidth=2, marker='o')
plt.xlabel('Mc', fontsize=18)
plt.ylabel('b Value (MLE)', fontsize=18)
plt.title('b - Mc', fontsize=20, color='r')
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(True, linestyle='--', linewidth=1.5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.)
ax.spines['left'].set_linewidth(1.)
ax.spines['top'].set_linewidth(1.)
ax.spines['right'].set_linewidth(1.)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout()
plt.show()

print((time.time()-start_time)/60, 'minutes')

