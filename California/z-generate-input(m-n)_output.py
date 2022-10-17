import config

time = config.time
np = config.np
pd = config.pd

start_time = time.time()
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

span_lat = config.span_lat
span_lon = config.span_lon
time_window = config.time_window
next_month = config.next_month
blocks = config.blocks 
features = config.features
index = config.index

n_splits = config.n_splits
epochs = config.epochs
learning_rate = config.learning_rate
filename = config.filename
catolog_name = config.catolog_name

file_location = r'C:\Users\cshu\Desktop\shi\code'

dataset = pd.read_csv(file_location+r"\data\dataset-"+catolog_name+r".txt", 
    delimiter=' ', header=None, dtype=np.float32)
dataset = dataset.values
print('\n  initial dataset shape: ', dataset.shape)

print('\n  indicators: ')
print('\t', 'initial indicators shape: ', dataset[:, 1:].shape)
indicators = dataset[:, 1:].reshape(-1, blocks*features)
print('\t', 'reshape indicators=', indicators.shape)

names = globals()
global input_data_indicators
if config.m+next_month >= config.n:
    input_data_indicators = np.empty_like(indicators[next_month+config.m-1:, :])
    print('\t', 'input_data_indicators.shape =', input_data_indicators.shape)
    for i in np.arange(config.n):
        print('\t n=', i+next_month+config.m-1-(config.n-1), i-(config.n-1))  
        if i-(config.n-1) != 0:   
            names["input_data_indicators_"+str(i)] = \
                indicators[i+next_month+config.m-1-(config.n-1):i-(config.n-1), :]
        else:
            names["input_data_indicators_"+str(i)] = \
                indicators[i+next_month+config.m-1-(config.n-1):, :]  
        input_data_indicators = np.concatenate((input_data_indicators, 
            names["input_data_indicators_"+str(i)]), axis=1)
else:
    input_data_indicators = np.empty_like(indicators[next_month+config.n-1:, :])
    print('\t', 'input_data_indicators.shape =', input_data_indicators.shape)
    for i in np.arange(config.n):
        print('\t n=', i, i-(next_month+config.n-1))
        if i-(next_month+config.n-1) != 0:
            names["input_data_indicators_"+str(i)] = \
                indicators[i:i-(next_month+config.n-1), :] 
        else:
            names["input_data_indicators_"+str(i)] = \
                indicators[i:, :]
        input_data_indicators = np.concatenate((input_data_indicators, 
            names["input_data_indicators_"+str(i)]), axis=1)
input_data_indicators = input_data_indicators[:, indicators.shape[1]:]
print('\t', 'input_data_indicators.shape =', input_data_indicators.shape)

print('\n  最大震级M: ')
M = dataset[:, 0].reshape(-1, blocks)
print('\t', 'initial M.shape: ', M.shape)

global input_data_M
if config.m+next_month >= config.n:
    input_data_M = np.empty_like(M[next_month+config.m-1:, 0]).reshape(-1, 1)
    print('\t', 'input_data_M.shape =', input_data_M.shape)
    for i in np.arange(config.m):
        print('\t m =', i, i-(next_month+config.m-1))
        if i-(next_month+config.m-1) != 0:
            names["input_data_M_"+str(i)] = \
                M[i:i-(next_month+config.m-1), :] 
        else:
            names["input_data_M_"+str(i)] = \
                M[i:, :] 
        input_data_M = np.concatenate((input_data_M, names["input_data_M_"+str(i)]), axis=1)
else:
    input_data_M = np.empty_like(M[next_month+config.n-1:, 0]).reshape(-1, 1)
    print('\t', 'input_data_M.shape =', input_data_M.shape)
    for i in np.arange(config.m):
        print('\t m=', i+(config.n-1)-(config.m-1)-next_month-1, \
            i+(config.n-1)-(config.m-1)-next_month-1-(next_month+config.n-1))
        if i+(config.n-1)-(config.m-1)-next_month-(next_month+config.n) != 0:
            names["input_data_M_"+str(i)] = \
                M[i+(config.n-1)-(config.m-1)-next_month-1:\
                    i+(config.n-1)-(config.m-1)-next_month-1-(next_month+config.n-1), :] 
        else:
            names["input_data_M_"+str(i)] = \
                M[i+(config.n-1)-(config.m-1)-next_month-1:, :] 
        input_data_M = np.concatenate((input_data_M, names["input_data_M_"+str(i)]), axis=1)
input_data_M = input_data_M[:, 1:].reshape(-1, blocks*config.m)
print('\t', 'input_data_M =', input_data_M.shape)

input_data = np.concatenate((input_data_M, input_data_indicators), axis=1)

if config.m+next_month >= config.n:
    output_data = M[next_month+config.m-1:, :]
else:
    output_data = M[next_month+config.n-1:, :]

print('\n  input_data', input_data.shape, 'output_data', output_data.shape)

np.savetxt(file_location+r'\data\input_'+filename+r".txt", 
    input_data, fmt="%.4f", delimiter=" ")
np.savetxt(file_location+r'\data\output_'+filename+r".txt", 
    output_data, fmt="%.4f", delimiter=" ")
