import os
import numpy as np
from tqdm import tqdm
import torch.nn.utils
import argparse
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from acds.benchmarks import get_lorenz
import joblib

from acds.archetypes import RandomizedOscillatorsNetwork, DeepReservoir


# =========================================================
# Set arguments to pass from command line
# =========================================================
parser = argparse.ArgumentParser(description='training parameters')

# GENERAL PARAMETERS:
#   folder path where results are saved
parser.add_argument("--resultroot", type=str, default='experiments/my_stuff/trained_architectures')
#   suffix to add to results file
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
#   number of hidden unities in the net
parser.add_argument('--n_hid', type=int, default=6800, help='hidden size of recurrent net')
#   prediction lag
parser.add_argument("--lag", type=int, default=25, help="prediction lag")
#   force use cpu
parser.add_argument("--cpu", action="store_true")
#   using test set for evaluating the trained model
parser.add_argument("--use_test", action="store_true")
#   number of trials (how many times we want to run the experiment)
parser.add_argument("--trials", type=int, default=1, help="How many times to run the experiment")
#   model choice
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")

# PARAMETERS FOR ALL RONs MODELS:
#   stiffness (gamma)
parser.add_argument("--gamma", type=float, default=10.0, help="y controle parameter <gamma> of the coRNN")
parser.add_argument("--gamma_range", type=float, default=1.0, help="y controle parameter <gamma> of the coRNN")
#   damping (epsilon)
parser.add_argument("--epsilon", type=float, default=10.0, help="z controle parameter <epsilon> of the coRNN")
parser.add_argument("--epsilon_range", type=float, default=0.5, help="z controle parameter <epsilon> of the coRNN")

# PARAMETERS FOR ESN MODEL:
#   leaky factor
parser.add_argument("--leaky", type=float, default=0.0001)

# OTHER PARAMETERS
#   temporal discretization step (dt)
parser.add_argument("--dt", type=float, default=0.17, help="step size <dt> of the coRNN")
#   spectral radius (max abs eigenvalue of the recurrent matrix). For ESN and pure RON
parser.add_argument("--rho", type=float, default=0.99, help="spectral radius")
#   input scaling (max abs value of the input-reservoir connection weights)
parser.add_argument("--inp_scaling", type=float, default=0.1, help="ESN input scaling")
#   topology of the reservoir (and scaling factor for ring/band/toeplitz cases). For pure RON
parser.add_argument("--topology", 
                    type=str, 
                    default="full", 
                    choices=["full", "ring", "band", "lower", "toeplitz", "orthogonal", "antisymmetric"], 
                    help="Topology of the reservoir")
#   sparsity of the connections in the reservoir (0: fully connected; 1: everything unconnected). For ESN and pure RON
parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity of the reservoir")


args = parser.parse_args()


# =========================================================
# Preparation
# =========================================================

# Choose device (CPU/GPU)
device = (torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)
print("Using device ", device)

# Prepare parameters
n_inp = 5 # multidimensional time sequence
n_out = 5 # multidimensional time sequence
washout = 200 # whashout steps
lag = args.lag

gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

# Function to test the trained classifier
@torch.no_grad()
def test(dataset, classifier, scaler):
    target = dataset[:, (lag+washout):].numpy().reshape(-1, n_inp) # from k=Nw+Nl to k=N-1. Shape (B*(N-Nw-Nl), n_inp)
    datapoints = dataset[:, :(2000+washout)].to(device) # datapoints, from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_inp)
    out = model(datapoints)[0].cpu().numpy() # reservoir response from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_hid)
    activations = out[:, washout:] # activations, from k=Nw to k=N-Nl-1. Shape (B, N-Nw-Nl, n_hid)
    activations = activations.reshape(-1, args.n_hid) # shape (B, N-Nw-Nl, n_hid) -> (B*(N-Nw-Nl), n_hid)
    activations = scaler.transform(activations)
    predictions = classifier.predict(activations)
    mse = np.mean(np.square(predictions - target))
    rmse = np.sqrt(mse)
    norm = np.sqrt(np.square(target).mean())
    nrmse = rmse / (norm + 1e-9)
    return nrmse


# =========================================================
# Run the experiment the desired number of times
# =========================================================

# Create folder for saving the trained networks
if args.ron:
    netw = 'RON'
elif args.esn:
    netw = 'ESN'
else:
    raise ValueError("Wrong model choice.")

suffix = f"_{args.resultsuffix}" if args.resultsuffix else ""
if args.ron:
    save_dir = os.path.join(args.resultroot, f'lorenz_{netw}_{args.topology}{suffix}')
else:
    save_dir = os.path.join(args.resultroot, f'lorenz_{netw}{suffix}')
    
os.makedirs(save_dir, exist_ok=True)  # create folder if not there already

# Iterations
train_nrmse_list, valid_nrmse_list, test_nrmse_list = [], [], []
for i in tqdm(range(args.trials), 'Trials', leave=False):
    # Initialize the model
    print('\nInitializing the model...')
    if args.ron:
        model = RandomizedOscillatorsNetwork(
            n_inp = n_inp,
            n_hid = args.n_hid,
            dt = args.dt,
            gamma = gamma,
            epsilon = epsilon,
            rho = args.rho,
            input_scaling = args.inp_scaling,
            topology = args.topology,
            device=device,
        ).to(device)
    elif args.esn:
        model = DeepReservoir(
            input_size=n_inp,
            tot_units=args.n_hid,
            input_scaling=args.inp_scaling,
            spectral_radius=args.rho,
            leaky=args.leaky,
            connectivity_recurrent=int((1 - args.sparsity) * args.n_hid),
            connectivity_input=args.n_hid,
            dt = args.dt,
        ).to(device)

    # Build datasets
    print('\nBuilding datasets...')
    train_dataset = get_lorenz(dim=n_inp, num_batch=128, F=8, lag=lag, washout=washout) # from k=0 to k=N-1. Shape (B, N, n_inp)
    valid_dataset = get_lorenz(dim=n_inp, num_batch=128, F=8, lag=lag, washout=washout) # from k=0 to k=N-1. Shape (B, N, n_inp)
    test_dataset = get_lorenz(dim=n_inp, num_batch=128, F=8, lag=lag, washout=washout) # from k=0 to k=N-1. Shape (B, N, n_inp)

    train_sequence = train_dataset[:, :-lag].to(device) # from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_inp)
    target = train_dataset[:, (lag+washout):].numpy() # from k=Nw+Nl to k=N-1. Shape (B, N-Nw-Nl, n_inp)
    target = target.reshape(-1, n_inp) # shape (B, N-Nw-Nl, n_inp) -> (B*(N-Nw-Nl), n_inp). Merge the first 2 dimensions (batches and timesteps)

    # Train the output layer (1): pass the train input sequence to the model
    print('\nGenerating activations for training...')
    with torch.no_grad():
        out = model(train_sequence) # forward method gives a tuple with 2 elements...
    out = out[0].cpu().numpy() # ...we only want the first one, which is a (B, N-Nl, n_hid) array. It's the reservoir's states evolution from k=0 to k=N-Nl-1
    activations = out[:, washout:] # remove the initial washout steps. Shape (B, N-Nl-Nw, n_hid). It's the reservoir's states evolution from k=Nw to k=N-Nl-1
    activations = activations.reshape(-1, args.n_hid) # shape (B, N-Nl-Nw, n_hid) -> (B*(N-Nl-Nw), n_hid)
    
    # Train the output layer (2): logistic regression of the output layer
    print('\nTraining the output layer (regression)...')
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    predictor = Ridge(max_iter=1000).fit(activations, target)
    
    # Evaluate the performances of the trained classifier
    print('\nEvaluating perfomances...')
    train_nrmse = test(train_dataset, predictor, scaler)
    valid_nrmse = test(valid_dataset, predictor, scaler) if not args.use_test else 0.0
    test_nrmse = test(test_dataset, predictor, scaler) if args.use_test else 0.0

    train_nrmse_list.append(train_nrmse)
    valid_nrmse_list.append(valid_nrmse)
    test_nrmse_list.append(test_nrmse)

    # Save the trained network
    print('\nSaving trained network...')
    if args.ron:
        model_path = os.path.join(save_dir, f"lorenz_{netw}_{args.topology}{suffix}_model_{i}.pt")
        torch.save(model.state_dict(), model_path) # save reservoir random parameters (epsilon, gamma, h2h, x2h, bias)
        scaler_path = os.path.join(save_dir, f"lorenz_{netw}_{args.topology}{suffix}_scaler_{i}.pkl")
        joblib.dump(scaler, scaler_path) # save scaler
        classifier_path = os.path.join(save_dir, f"lorenz_{netw}_{args.topology}{suffix}_classifier_{i}.pkl")
        joblib.dump(predictor, classifier_path) # save classifier
    else:
        model_path = os.path.join(save_dir, f"lorenz_{netw}{suffix}_model_{i}.pt")
        torch.save(model.state_dict(), model_path) # save reservoir
        scaler_path = os.path.join(save_dir, f"lorenz_{netw}{suffix}_scaler_{i}.pkl")
        joblib.dump(scaler, scaler_path) # save scaler
        classifier_path = os.path.join(save_dir, f"lorenz_{netw}{suffix}_classifier_{i}.pkl")
        joblib.dump(predictor, classifier_path) # save classifier
    print()

# Save results
print('Saving results...')
if args.ron:
    f = open(os.path.join(save_dir, f"lorenz_log_{netw}_{args.topology}{suffix}.txt"), "a")
else:
    f = open(os.path.join(save_dir, f"lorenz_log_{netw}{suffix}.txt"), "a")

ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_acc, 3)) for train_acc in train_nrmse_list]} "
    f"valid: {[str(round(valid_acc, 3)) for valid_acc in valid_nrmse_list]} "
    f"test: {[str(round(test_acc, 3)) for test_acc in test_nrmse_list]}"
    f"mean/std train: {np.mean(train_nrmse_list), np.std(train_nrmse_list)} "
    f"mean/std valid: {np.mean(valid_nrmse_list), np.std(valid_nrmse_list)} "
    f"mean/std test: {np.mean(test_nrmse_list), np.std(test_nrmse_list)}"
)
f.write(ar + "\n")
f.close()
print('\nDone!')
