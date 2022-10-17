import config
import os

n_splits = config.n_splits

# os.system('python ./merge_catalog.py')
# os.system('python ./plot_before_main.py')

# os.system('python ./filter_catalog_California.py')
os.system('python ./plot_mc.py') # python plot-mc.py
os.system('python ./plot-cumu.py')

# os.system('python ./generate-seismic_factor1.py')
# os.system('python ./generate-seismic_factor.py')
# os.system('python ./plot-seismic_factor.py')

# if n_splits == 1:
#     print('\n\tsplit')
#     os.system('python ./lstm-split.py')
# else:
#     print('\n\tfold')
#     os.system('python ./lstm-fold.py')  
# os.system('python ./plot-all.py')
# os.system('python ./plot-loss.py')

