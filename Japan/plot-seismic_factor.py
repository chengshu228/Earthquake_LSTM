import config

time = config.time
plt = config.plt
np = config.np
pd = config.pd
tf = config.tf
matplotlib = config.matplotlib

span_lat = config.span_lat
span_lon = config.span_lon
time_window = config.time_window
next_month = config.next_month
blocks = config.blocks 
features = config.features
index = config.index
m = config.m
n = config.n
n_splits = config.n_splits
split_ratio = config.split_ratio
epochs = config.epochs
learning_rate = config.learning_rate
filename = config.filename
catolog_name = config.catolog_name
file_location = config.file_location
matplotlib.rcParams['axes.unicode_minus'] = False
fontcn = config.fontcn 
fonten = config.fonten

start_time = time.time()

factor = pd.read_csv(file_location+r"\factor\factor-"+catolog_name+r".txt", 
    delimiter=' ', header=None, dtype=np.float32)
factor = factor.values
print('\n  initial factor shape: ', factor.shape)
output_data = factor[:, 0].reshape(-1, blocks)
factor = np.concatenate((
    factor[:, 1].reshape(-1, 1),  # frequency
    factor[:, 2].reshape(-1, 1),  # max_magnitude
    factor[:, 3].reshape(-1, 1),  # mean_magnitude
    factor[:, 4].reshape(-1, 1), # b_lstsq
    factor[:, 5].reshape(-1, 1),  # b_mle
    factor[:, 6].reshape(-1, 1), # a_lstsq
    factor[:, 7].reshape(-1, 1), # max_mag_absence
    factor[:, 8].reshape(-1, 1), # rmse_lstsq
    factor[:, 9].reshape(-1, 1), # total_energy_square
    factor[:, 10].reshape(-1, 1), # mean_lon
    factor[:, 11].reshape(-1, 1), # rmse_lon
    factor[:, 12].reshape(-1, 1), # mean_lat
    factor[:, 13].reshape(-1, 1), # rmse_lat
    factor[:, 14].reshape(-1, 1), # k
    factor[:, 15].reshape(-1, 1), # epicenter_longitude
    factor[:, 16].reshape(-1, 1), # epicenter_latitude
), axis=1)
features = factor.shape[1]
input_data = factor.reshape(-1, blocks*(features*n+m))
print('\n  input_data', input_data.shape)

location_block = [0, 2, 18, 20]
linestyles = [':', '-', '--']
labels = []
for i in location_block:
    labels.append('block{}'.format(i+1))
colors = ['blueviolet', 'green', 'blue', 'goldenrod', 'cyan']
markers = ['p', 'd', 'v', '^', 'x', 'o', '+', '<', '>', 's', '*', 'P']
title =['frequency', 'max_magnitude', 'mean_magnitude', \
        'b_lstsq', 'b_mle', 'a_lstsq', \
        'max_mag_absence', 'rmse_lstsq', 'total_energy_square', \
        'mean_lon', 'rmse_lon', 'mean_lat', 'rmse_lat', \
        'k', 'epicenter_longitude', 'epicenter_latitude']

for earthquake_index in np.arange(0, features, 1):
    fig = plt.figure(figsize=(14, 6))
    fig.add_subplot(1,1,1)
    plt.title(f'{title[earthquake_index]}', fontdict=fonten, fontsize=20, color='red')
    plt.grid(True, linestyle='--', linewidth=1)
    for key, value in enumerate(location_block):
        print(earthquake_index+(features*value))
        plt.plot(np.arange(len(input_data)), 
            input_data[:, earthquake_index+(features*value)], 
            linewidth=2, marker=markers[key], label=labels[key], color=colors[key])   
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)  
        plt.ylabel(f'{title[earthquake_index]}', fontdict=fonten, fontsize=18)  
        plt.legend(loc='upper left', fontsize=12) 
        plt.xticks(size=14)
        plt.yticks(size=14)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)   
        ax.xaxis.set_major_locator(plt.MultipleLocator(24))
        plt.tight_layout()
    # plt.savefig(r".\figure\{}-Histgram-test.png".format(filename))
    plt.show()

