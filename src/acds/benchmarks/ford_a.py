import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F


########################################

class datasetforRC(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding gives problems with scikit-learn LogisticRegression of RC models
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)
    

class FordA_dataset(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding targets
        idx_targ = F.one_hot(idx_targ.type(torch.int64), num_classes=2).float()
        # reshape target for torch (batch, classes)
        idx_targ = idx_targ.reshape(idx_targ.shape[1])
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)

########################################

def get_forda_data(datasetpath, bs_train, bs_test, whole_train=False, RC=True):
    """
    Args
    ----
    datasetpath
        Path to the folder containing FordA data (FordA_TRAIN.txt and ForsA_TEST.txt files).
    bs_train, bs_test
        Batch sizes for train and test set in dataloaders.
    whole_train : bool
        If True, uses the whole FordA_TRAIN.txt for training. Otherwise splits it in 
        train/validation (default: False).
    RC : bool
        If True, labels as 0 or 1. If False, labels as [1,0] or [0,1] (one-shot encoding).

    Returns
    -------
    mytrainloader, myvalidloader, mytestloader
        Train, validation and test dataloaders. Each contains batches of pairs (x_batch, y_batch),
        where x_batch is a batch of data points (temporal sequences of length L) with shape
        (batch_size, L, 1), while y_batch is a batch of corresponding labels.
    """
    def fromtxt_to_numpy(filepath, valid_len=1320):
        # read the txt file
        forddata = np.genfromtxt(filepath, dtype='float64')
        # create a list of lists with each line of the txt file
        l = []
        for i in forddata:
            el = list(i)
            while len(el) < 3:
                el.append('a')
            l.append(el)
        # create a numpy array from the list of lists
        arr = np.array(l)
        if valid_len is None:
            test_targets = (arr[:,0]+1)/2
            test_series = arr[:,1:]
            return test_series, test_targets
        else:
            if valid_len == 0:
                train_targets = (arr[:,0]+1)/2
                train_series = arr[:,1:]
                val_targets = arr[0:0,0] # empty
                val_series = arr[0:0,1:] # empty
            elif valid_len > 0 :
                train_targets = (arr[:-valid_len,0]+1)/2
                train_series = arr[:-valid_len,1:]
                val_targets = (arr[-valid_len:,0]+1)/2
                val_series = arr[-valid_len:,1:]
            return train_series, train_targets, val_series, val_targets

    # Generate list of input-output pairs
    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            sample = (data_x[i,:], data_y[i])
            mydata.append(sample)
        return mydata

    # generate torch datasets
    train_file = datasetpath+'/FordA_TRAIN.txt'
    test_file = datasetpath+'/FordA_TEST.txt'
    if whole_train:
        valid_len = 0
    else:
        valid_len = 1320
    train_series, train_targets, val_series, val_targets = fromtxt_to_numpy(filepath=train_file, valid_len=valid_len)
    mytraindata, myvaldata = inp_out_pairs(train_series, train_targets), inp_out_pairs(val_series, val_targets)
    if RC:
        mytraindata, myvaldata = datasetforRC(mytraindata), datasetforRC(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filepath=test_file, valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = datasetforRC(mytestdata)
    else:
        mytraindata, myvaldata = FordA_dataset(mytraindata), FordA_dataset(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filepath=test_file, valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = FordA_dataset(mytestdata)


    # generate torch dataloaders
    mytrainloader = data.DataLoader(mytraindata,
                    batch_size=bs_train, shuffle=True, drop_last=True)
    myvaloader = data.DataLoader(myvaldata,
                        batch_size=bs_test, shuffle=False, drop_last=True)
    mytestloader = data.DataLoader(mytestdata,
                batch_size=bs_test, shuffle=False, drop_last=True)
    return mytrainloader, myvaloader, mytestloader