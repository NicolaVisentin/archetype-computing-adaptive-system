import argparse
import warnings
import os
import numpy as np
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks import get_mackey_glass


# =========================================================
# Set arguments to pass from command line
# =========================================================
parser = argparse.ArgumentParser(description="training parameters")

# GENERAL PARAMETERS:
#   folder path where Mackey-Glass data are downloaded
parser.add_argument("--dataroot", type=str, default='src/acds/benchmarks/raw', help="Path to the folder containing the mackey_glass.csv dataset")
#   folder path where results are saved
parser.add_argument("--resultroot", type=str, default='experiments/my_stuff/trained_architectures')
#   suffix to add to results file
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
#   number of hidden unities in the net
parser.add_argument("--n_hid", type=int, default=1000, help="hidden size of recurrent net")
#   batch size
parser.add_argument("--batch", type=int, default=30, help="batch size")
#   prediction lag
parser.add_argument("--lag", type=int, default=84, help="prediction lag")
#   force use cpu
parser.add_argument("--cpu", action="store_true")
#   model choice
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")
parser.add_argument("--pron", action="store_true")
parser.add_argument("--mspron", action="store_true")
#   input scaling (max abs value of the input-reservoir connection weights)
parser.add_argument("--inp_scaling", type=float, default=10.0, help="ESN input scaling")
#   using test set for evaluating the trained model
parser.add_argument("--use_test", action="store_true")
#   number of trials (how many times we want to run the experiment)
parser.add_argument("--trials", type=int, default=1, help="How many times to run the experiment")

# PARAMETERS FOR ALL RONs MODELS:
#   temporal discretization step (dt)
parser.add_argument("--dt", type=float, default=0.17, help="step size <dt> of the coRNN")
#   stiffness (gamma)
parser.add_argument("--gamma", type=float, default=2.0, help="y controle parameter <gamma> of the coRNN")
parser.add_argument("--gamma_range", type=float, default=1.0, help="y controle parameter <gamma> of the coRNN")
#   damping (epsilon)
parser.add_argument("--epsilon", type=float, default=2.0, help="z controle parameter <epsilon> of the coRNN")
parser.add_argument("--epsilon_range", type=float, default=0.5, help="z controle parameter <epsilon> of the coRNN")

# PARAMETERS FOR ESN MODEL:
#   leaky factor
parser.add_argument("--leaky", type=float, default=1.0)

# OTHER SPECIFIC PARAMETERS
#   spectral radius (max abs eigenvalue of the recurrent matrix). For ESN and pure RON
parser.add_argument("--rho", type=float, default=0.9, help="ESN spectral radius")
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

assert args.dataroot is not None, "No dataroot provided"
if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()
assert os.path.exists(args.resultroot), \
    f"{args.resultroot} folder does not exist, please create it and run the script again."
assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"


# =========================================================
# Preparation
# =========================================================

# Choose device (CPU/GPU)
device = (torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)

# Prepare parameters
n_inp = 1 # monodimensional time sequence
n_out = 1 # monodimensional time sequence
washout = 200 # whashout steps

gamma = (args.gamma - args.gamma_range / 2.0, 
         args.gamma + args.gamma_range / 2.0)
epsilon = (args.epsilon - args.epsilon_range / 2.0, 
           args.epsilon + args.epsilon_range / 2.0)

# Function to test the trained classifier
criterion_eval = torch.nn.L1Loss()
@torch.no_grad()
def test(dataset, target, classifier, scaler):
    dataset = dataset.reshape(1, -1, 1).to(device)
    target = target.reshape(-1, 1).numpy()
    activations = model(dataset)[0].cpu().numpy()
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid)
    activations = scaler.transform(activations)
    predictions = classifier.predict(activations)
    error = criterion_eval(torch.from_numpy(predictions).float(), torch.from_numpy(target.squeeze()).float()).item()
    return error


# =========================================================
# Run the experiment the desired number of times
# =========================================================

# Create folder for saving the trained networks
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

suffix = f"_{args.resultsuffix}" if args.resultsuffix else ""
if args.ron:
    save_dir = os.path.join(args.resultroot, f'MG_{netw}_{args.topology}{suffix}')
else:
    save_dir = os.path.join(args.resultroot, f'MG_{netw}{suffix}')
    
os.makedirs(save_dir, exist_ok=True)  # create folder if not there already

# Iterations
train_mse, valid_mse, test_mse = [], [], []
for i in tqdm(range(args.trials), 'Trials', leave=False):
    # Initialize the model
    print('\nInitializing the model...')
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
            device=device
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
        raise ValueError("Wrong model name")

    # Build datasets
    print('\nBuilding datasets...')
    (
        (train_dataset, train_target),
        (valid_dataset, valid_target),
        (test_dataset, test_target),
    ) = get_mackey_glass(csvfolder=args.dataroot, lag=args.lag, washout=washout)

    train_sequence = train_dataset.reshape(1, -1, 1).to(device) # shape (N-Nl,) -> (1, N-Nl, 1)
    target = train_target.reshape(-1, 1).numpy() # shape (N-Nl-Nw,) -> (N-Nl-Nw, 1)

    # Train the output layer (1): pass the train input sequence to the model
    print('\nGenerating activations for training...')
    output = model(train_sequence) # forward method gives a tuple with 2 elements...
    output = output[0].cpu().numpy() # ...we only want the first one, which is a (1, N-Nl, n_hid) array. It's the reservoir's states evolution from k=0 to k=N-Nl-1
    activations = output[:, washout:] # remove the initial washout steps. Shape (1, N-Nl-Nw, n_hid). It's the reservoir's states evolution from k=Nw to k=N-Nl-1
    activations = activations.reshape(-1, args.n_hid) # shape (1, N-Nl-Nw, n_hid) -> (N-Nl-Nw, n_hid)

    # Train the output layer (2): logistic regression of the output layer
    print('\nTraining the output layer (regression)...')
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = Ridge(max_iter=1000).fit(activations, target)

    # Evaluate the performances of the trained classifier
    print('\nEvaluating perfomances...')
    train_nmse = test(train_dataset, train_target, classifier, scaler) # on the train set
    valid_nmse = test(valid_dataset, valid_target, classifier, scaler) if not args.use_test else 0.0 # on the validation set
    test_nmse = test(test_dataset, test_target, classifier, scaler) if args.use_test else 0.0 # on the test set
    train_mse.append(train_nmse)
    valid_mse.append(valid_nmse)
    test_mse.append(test_nmse)

    # Save the trained network
    print('\nSaving trained network...')
    if args.ron:
        model_path = os.path.join(save_dir, f"MG_{netw}_{args.topology}{suffix}_model_{i}.pt")
        torch.save(model.state_dict(), model_path) # save reservoir random parameters (epsilon, gamma, h2h, x2h, bias)
        scaler_path = os.path.join(save_dir, f"MG_{netw}_{args.topology}{suffix}_scaler_{i}.pkl")
        joblib.dump(scaler, scaler_path) # save scaler
        classifier_path = os.path.join(save_dir, f"MG_{netw}_{args.topology}{suffix}_classifier_{i}.pkl")
        joblib.dump(classifier, classifier_path) # save classifier
    else:
        model_path = os.path.join(save_dir, f"MG_{netw}{suffix}_model_{i}.pt")
        torch.save(model.state_dict(), model_path) # save reservoir
        scaler_path = os.path.join(save_dir, f"MG_{netw}{suffix}_scaler_{i}.pkl")
        joblib.dump(scaler, scaler_path) # save scaler
        classifier_path = os.path.join(save_dir, f"MG_{netw}{suffix}_classifier_{i}.pkl")
        joblib.dump(classifier, classifier_path) # save classifier
    print()

# Save results
print('Saving results...')
if args.ron:
    f = open(os.path.join(save_dir, f"MG_log_{netw}_{args.topology}{suffix}.txt"), "a")
else:
    f = open(os.path.join(save_dir, f"MG_log_{netw}{suffix}.txt"), "a")

ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_acc, 2)) for train_acc in train_mse]} "
    f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_mse]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_mse]}"
    f"mean/std train: {np.mean(train_mse), np.std(train_mse)} "
    f"mean/std valid: {np.mean(valid_mse), np.std(valid_mse)} "
    f"mean/std test: {np.mean(test_mse), np.std(test_mse)}"
)
f.write(ar + "\n")
f.close()
print('\nDone!')
