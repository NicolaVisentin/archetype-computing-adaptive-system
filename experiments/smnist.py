import argparse
import os
import warnings
import numpy as np
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks import get_mnist_data


# =========================================================
# Set arguments to pass from command line
# =========================================================
parser = argparse.ArgumentParser(description="training parameters")

# GENERAL PARAMETERS:
#   folder path where MNIST data are downloaded
parser.add_argument("--dataroot", type=str)
#   folder path where results are saved
parser.add_argument("--resultroot", type=str)
#   suffix to add to results file
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
#   number of hidden unities in the net
parser.add_argument("--n_hid", type=int, default=256, help="hidden size of recurrent net")
#   batch size
parser.add_argument("--batch", type=int, default=1000, help="batch size")
#   force use cpu
parser.add_argument("--cpu", action="store_true")
#   model choice
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")
parser.add_argument("--pron", action="store_true")
parser.add_argument("--mspron", action="store_true")
#   input scaling (max abs value of the input-reservoir connection weights)
parser.add_argument("--inp_scaling", type=float, default=1.0, help="ESN input scaling")
#   using test set for evaluating the trained model
parser.add_argument("--use_test", action="store_true")
#   number of trials (how many times we want to run the experiment)
parser.add_argument("--trials", type=int, default=1, help="How many times to run the experiment")

# PARAMETERS FOR ALL RONs MODELS:
#   temporal discretization step (dt)
parser.add_argument("--dt", type=float, default=0.042, help="step size <dt> of the coRNN")
#   damping (gamma)
parser.add_argument("--gamma", type=float, default=2.7, help="y controle parameter <gamma> of the coRNN")
parser.add_argument("--gamma_range", type=float, default=2.7, help="y controle parameter <gamma> of the coRNN")
#   stiffness (epsilon)
parser.add_argument("--epsilon", type=float, default=4.7, help="z controle parameter <epsilon> of the coRNN")
parser.add_argument("--epsilon_range", type=float, default=4.7, help="z controle parameter <epsilon> of the coRNN")

# PARAMETERS FOR ESN MODEL:
#   spectral radius (max abs eigenvalue of the recurrent matrix)
parser.add_argument("--rho", type=float, default=0.99, help="ESN spectral radius")
#   leaky factor
parser.add_argument("--leaky", type=float, default=1.0)

# OTHER SPECIFIC PARAMETERS
#   diffusive term (to ensure stability of the forward Euler method). For pure RON
parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")
#   topology of the reservoir (and scaling factor for ring/band/toeplitz cases). For pure RON
parser.add_argument("--topology", 
                    type=str, 
                    default="full", 
                    choices=["full", "ring", "band", "lower", "toeplitz", "orthogonal", "antisymmetric"], 
                    help="Topology of the reservoir")
parser.add_argument("--reservoir_scaler", type=float, default=1.0, help="Scaler in case of ring/band/toeplitz reservoir")
#   sparsity of the connections in the reservoir (0: fully connected; 1: everything unconnected). For ESN and pure RON
parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity of the reservoir")

args = parser.parse_args()
if args.dataroot is None:
    warnings.warn("No dataroot provided. Using current location as default.")
    args.dataroot = os.getcwd()
if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()
assert os.path.exists(args.resultroot), \
    f"{args.resultroot} folder does not exist, please create it and run the script again."
assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"


# =========================================================
# Preparation
# =========================================================

# # Visualize one datum (image) with its label (just a check)
# _, _, test_loader = get_mnist_data(
#     root=args.dataroot, 
#     bs_train=args.batch, 
#     bs_test=args.batch
# )
# data_batch = iter(test_loader)    # get one batch of data from test set
# images, labels = next(data_batch) # divide images and labels from the batch
# print(f'Batch size: {images.shape[0]}\n'
#       f'Single datum dimension (image (C,W,H)): {images[0].numpy().shape}\n'
# )
# plt.figure()
# plt.imshow(images[0].squeeze(), cmap='gray')
# plt.title(f'Label: {labels[0]}')
# plt.show()
# exit()

# Function to test the trained classifier
@torch.no_grad()
def test(data_loader, classifier, scaler):
    activations, ys = [], []
    # iterate through batches
    for images, labels in tqdm(data_loader, f'Testing the model', leave=False):
        images = images.to(device)
        images = images.view(images.shape[0], -1) # (batch_size, 1, 28, 28) --> (batch_size, 784)
        images = images.unsqueeze(-1)             # (batch_size, 784) --> (batch_size, 784, 1) as the forward 
                                                  # method of the model expects (batch_size, num_timesteps, input_dim)
        output = model(images) # forward method gives a tuple with 2 elements...
        output = output[-1]    # ...we only want the last one, which is a list...
        output = output[0]     # ...form which we extract the first element: a tensor (batch_size, n_hidden). Each row (associated
                               # with one element of the batch) contains the last hidden states for all the hidden units
        activations.append(output.cpu())
        ys.append(labels)

    activations = torch.cat(activations, dim=0).numpy() # shape (num_train_images, num_hidden_units)
    activations = scaler.transform(activations)        
    ys = torch.cat(ys, dim=0).numpy()                   # shape (num_train_images,)

    return classifier.score(activations, ys)

# Choose device (CPU/GPU)
device = (torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)

# Prepare parameters
n_inp = 1  # 1 input (sequence of pixels)
n_out = 10 # 10 classes ({0,1,2,3,4,5,6,7,8,9})

gamma = (args.gamma - args.gamma_range / 2.0, 
         args.gamma + args.gamma_range / 2.0)
epsilon = (args.epsilon - args.epsilon_range / 2.0, 
           args.epsilon + args.epsilon_range / 2.0)


# =========================================================
# Run the experiment the desired number of times
# =========================================================

train_accs, valid_accs, test_accs = [], [], []
for i in tqdm(range(args.trials), 'Trials', leave=False):
    # Initialize the model
    print('Initializing the model...')
    if args.esn:
        model = DeepReservoir(
            input_size=n_inp,
            tot_units=args.n_hid,
            input_scaling=args.inp_scaling,
            spectral_radius=args.rho,
            leaky=args.leaky,
            connectivity_recurrent=int((1 - args.sparsity) * args.n_hid),
            connectivity_input=args.n_hid,
        ).to(device)
    elif args.ron:
        model = RandomizedOscillatorsNetwork(
            n_inp=n_inp,
            n_hid=args.n_hid,
            dt=args.dt,
            gamma=gamma,
            epsilon=epsilon,
            diffusive_gamma=args.diffusive_gamma,
            rho=args.rho,
            input_scaling=args.inp_scaling,
            topology=args.topology,
            reservoir_scaler=args.reservoir_scaler,
            sparsity=args.sparsity,
            device=device,
        ).to(device)
    elif args.pron:
        model = PhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.inp_scaling,
            device=device,
        ).to(device)
    elif args.mspron:
        model = MultistablePhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.inp_scaling,
            device=device
        ).to(device)
    else:
        raise ValueError("Wrong model choice.")
    print('Model initialized!')

    # Build dataloaders for MNIST task
    print('Building datasets...')
    train_loader, valid_loader, test_loader = get_mnist_data(
        root=args.dataroot, 
        bs_train=args.batch, 
        bs_test=args.batch
    )
    print('Datasets built!')

    # Train the output layer (classifier) (1): pass all the inputs in the train set to the model
    print('Generating previsions for training...')
    activations, ys = [], []
    for images, labels in tqdm(train_loader, 'Model forward', leave=False):
        images = images.to(device)
        images = images.view(images.shape[0], -1) # (batch_size, 1, 28, 28) --> (batch_size, 784)
        images = images.unsqueeze(-1)             # (batch_size, 784) --> (batch_size, 784, 1) as the forward method of the model expects (batch_size, num_timesteps, input_dim)
        output = model(images) # forward method gives a tuple with 2 elements...
        output = output[-1]    # ...we only want the last one, which is a list...
        output = output[0]     # ...form which we extract the first element: a tensor (batch_size, n_hidden). Each row (associated
                               # with one element of the batch) contains the last hidden states for all the hidden units
        activations.append(output.cpu())
        ys.append(labels)
        
    activations = torch.cat(activations, dim=0).numpy() # shape (num_train_images, num_hidden_units)
    ys = torch.cat(ys, dim=0).squeeze().numpy()         # shape (num_train_images,)
    print('Previsions generated!')

    # Train the output layer (classifier) (2): logistic regression of the output layer
    print('Training the classifier (regression)...')
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)
    print('Training finished!')

    # Evaluate the performances of the trained classifier
    print('Evaluating perfomances...')
    train_acc = test(train_loader, classifier, scaler)                               # on the training set
    valid_acc = test(valid_loader, classifier, scaler) if not args.use_test else 0.0 # on the validation set
    test_acc = test(test_loader, classifier, scaler) if args.use_test else 0.0       # on the testing set
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)
    print('Testing finished!')

    print('Saving trained network...')
    save_dir = os.path.join(args.resultroot, 'trained_architectures')
    os.makedirs(save_dir, exist_ok=True)  # create folder if not there already
    if args.ron:
        netw = 'RON'
    elif args.pron:
        netw = 'PRON'
    elif args.mspron:
        netw = 'MSPRON'
    elif args.esn:
        netw = 'ESN'
    else:
        raise ValueError("Wrong model choice.")

    model_path = os.path.join(save_dir, f"sMNIST_{netw}_{args.topology}{args.resultsuffix}_model_{i}.pt")
    torch.save(model.state_dict(), model_path)

    scaler_path = os.path.join(save_dir, f"sMNIST_{netw}_{args.topology}{args.resultsuffix}_scaler_{i}.pkl")
    joblib.dump(scaler, scaler_path)

    classifier_path = os.path.join(save_dir, f"sMNIST_{netw}_{args.topology}{args.resultsuffix}_classifier_{i}.pkl")
    joblib.dump(classifier, classifier_path)
    print('Network saved!')

# Save results
print('Saving results...')
if args.ron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_RON_{args.topology}{args.resultsuffix}.txt"), "a")
elif args.pron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_PRON{args.resultsuffix}.txt"), "a")
elif args.mspron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_MSPRON{args.resultsuffix}.txt"), "a")
elif args.esn:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_ESN{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")

ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_acc, 2)) for train_acc in train_accs]} "
    f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_accs]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_accs]}"
    f"mean/std train: {np.mean(train_accs), np.std(train_accs)} "
    f"mean/std valid: {np.mean(valid_accs), np.std(valid_accs)} "
    f"mean/std test: {np.mean(test_accs), np.std(test_accs)}"
)
f.write(ar + "\n")
f.close()
print('Done!')
