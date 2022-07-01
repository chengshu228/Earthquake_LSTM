import config

os = config.os
n_splits = config.n_splits

# os.system('python ./filter_catalog_chuandian.py')
# os.system('python ./generate-seismic_factor.py')

if n_splits == 1:
    print('\n\t split')
    os.system('python ./lstm-split.py')
else:
    print('\n\t fold')
    os.system('python ./lstm-fold.py')
    
os.system('python ./plot-all.py')

# os.system('python ./plot-loss.py')
# os.system('python ./plot-b-value.py')
# os.system('python ./plot-cumu.py')
# os.system('python ./plot-seismic_factor.py')
