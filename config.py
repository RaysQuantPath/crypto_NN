from pandas import Timestamp

run1 = {
    'folder': 'asset_data/raw_data_15_min/',
    # 'folder' : 'raw_data_1_hour/',
    # 'folder' : 'raw_data_30_min/',
    #'folder' : 'raw_data_1_day/',

    'reports': 'reports/',
    'alpha': 0.004,  # computed in determine_alpha.py
    'beta': 0.045,  # ignore sample greater than beta in percent of change
    'seed': 114514,
    'commission fee': 0.0001,  # 0.0004,  # 0.001,

    'b_window': 30,
    'f_window': 1,

    # used in define the grid for searching backward and forward window
    'b_lim_sup_window': 31,
    'f_lim_sup_window': 2,
    
    'back_test_start': Timestamp("2020-01-01"),
    'back_test_end': Timestamp("2021-04-04"),
    'suffix': 'ncr',

    'stop_loss': 0.25,
    
    'off_label_set': [], #['BTCUSDT', 'ETHUSDT', 'AXLUSDT'],  # list of coin to be excluded from training/test set. Used in backtesting

    'balance_algo': 'srs',  # 'ncr', 'srs', None
    'loss_func': 'categorical',  # 'focal', 'categorical'
    
    'epochs': 12  # how many epochs spent in training neural network
}

RUN = run1
