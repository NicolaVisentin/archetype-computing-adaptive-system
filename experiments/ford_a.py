import torch
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
from acds.archetypes import RandomizedOscillatorsNetwork
from acds.benchmarks import get_forda_data


# =========================================================
# Set arguments to pass from command line
# =========================================================
parser = argparse.ArgumentParser(description='training parameters')

# GENERAL PARAMETERS:
#   folder path where FordA data are downloaded
parser.add_argument("--dataroot", type=str, default='src/acds/benchmarks/raw/FordA')
#   folder path where results are saved
parser.add_argument("--resultroot", type=str, default='experiments/my_stuff/trained_architectures')
#   suffix to add to results file
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
#   number of hidden unities in the net
parser.add_argument("--n_hid", type=int, default=100, help="hidden size of recurrent net")
#   batch size
parser.add_argument("--batch", type=int, default=120, help="batch size")
#   force use cpu
parser.add_argument("--cpu", action="store_true")
#   using test set for evaluating the trained model
parser.add_argument("--use_test", action="store_true")
#   number of trials (how many times we want to run the experiment)
parser.add_argument("--trials", type=int, default=1, help="How many times to run the experiment")
#   temporal discretization step (dt)
parser.add_argument("--dt", type=float, default=0.2, help="step size <dt> of the coRNN")

# PARAMETERS FOR ALL RON MODELS:
#   stiffness (gamma)
parser.add_argument("--gamma", type=float, default=1.0, help="y controle parameter <gamma> of the coRNN")
parser.add_argument("--gamma_range", type=float, default=0.5, help="y controle parameter <gamma> of the coRNN")
#   damping (epsilon)
parser.add_argument("--epsilon", type=float, default=5.1, help="z controle parameter <epsilon> of the coRNN")
parser.add_argument("--epsilon_range", type=float, default=5.0, help="z controle parameter <epsilon> of the coRNN")
#   input scaling (max abs value of the input-reservoir connection weights)
parser.add_argument("--inp_scaling", type=float, default=0.1, help="ESN input scaling")
#   spectral radius (max abs eigenvalue of the recurrent matrix). For ESN and pure RON
parser.add_argument("--rho", type=float, default=0.9, help="ESN spectral radius")


args = parser.parse_args()


# =========================================================
# Preparation
# =========================================================

# Function to test the model's accuracy
@torch.no_grad()
def test(data_loader, classifier, scaler):
    activations, ys = [], []
    # iterate through batches
    for x_batch, y_batch in tqdm(data_loader, f'Testing the model', leave=False):
        x_batch = x_batch.to(device) # shape (batch_size, L, 1)
        output_batch = model(x_batch)[-1][0] # last (in time) value for each hidden unit. Shape (batch_size, n_hidden)
        activations.append(output_batch.cpu())
        ys.append(y_batch)

    activations = torch.cat(activations, dim=0).numpy() # shape (test_size, num_hidden_units)
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy().ravel()

    return classifier.score(activations, ys)

# Choose device (CPU/GPU)
device = (torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)

# Prepare parameters
n_inp = 1 # 1 input (sequence of pixels)
n_out = 2 # binary classification

gamma = (args.gamma - args.gamma_range / 2.0, 
         args.gamma + args.gamma_range / 2.0)
epsilon = (args.epsilon - args.epsilon_range / 2.0, 
           args.epsilon + args.epsilon_range / 2.0)


# =========================================================
# Run the experiment the desired number of times
# =========================================================

# Create folder for saving the trained networks
netw = 'RON'
suffix = f"_{args.resultsuffix}" if args.resultsuffix else ""
save_dir = os.path.join(args.resultroot, f'FordA{netw}_full{suffix}')
os.makedirs(save_dir, exist_ok=True)  # create folder if not there already

# Iterations
train_accs, valid_accs, test_accs = [], [], []
for i in tqdm(range(args.trials), 'Trials', leave=False):
    accs = []
    # Initialize the model
    print('\nInitializing the model...')
    model = RandomizedOscillatorsNetwork(
            n_inp=n_inp,
            n_hid=args.n_hid,
            dt=args.dt,
            gamma=gamma,
            epsilon=epsilon,
            rho=args.rho,
            input_scaling=args.inp_scaling,
            device=device,
        ).to(device)
    
    # Build dataloaders
    print('\nBuilding datasets...')
    (
        train_loader, valid_loader, test_loader # torch dataloaders. For each batch, pair (x_batch, y_batch) where x_batch.shape=(batch_size,L,1) and y_batch.shape=(batch_size,1)
    ) = get_forda_data(
        datasetpath=args.dataroot,
        bs_train=args.batch,
        bs_test=args.batch,
        whole_train=True,
        RC=True
    )

    # Train the output layer (classifier) (1): pass all the inputs in the train set to the model
    print('\nGenerating previsions for training...')
    activations, ys = [], []
    for x_batch, y_batch in tqdm(train_loader, 'Model forward', leave=False):
        x_batch = x_batch.to(device) # shape (batch_size, L, 1)
        output_batch = model(x_batch) # forward method gives a tuple with 2 elements...
        output_batch = output_batch[-1] # ...we only want the last one, which is a list...
        output_batch = output_batch[0] # ...form which we extract the first element: a tensor (batch_size, n_hidden). Each row (associated
                                       # with one element of the batch) contains the last hidden states for all the hidden units
        activations.append(output_batch.cpu())
        ys.append(y_batch)

    activations = torch.cat(activations, dim=0).numpy() # shape (train_size, num_hidden_units)
    ys = torch.cat(ys, dim=0).numpy().ravel() # shape (train_size,)

    # Train the output layer (classifier) (2): logistic regression of the output layer
    print('\nTraining the classifier (regression)...')
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)

    # Evaluate the performances of the trained classifier
    print('\nEvaluating perfomances...')
    train_acc = test(train_loader, classifier, scaler)                               # on the training set
    valid_acc = test(valid_loader, classifier, scaler) if not args.use_test else 0.0 # on the validation set
    test_acc = test(test_loader, classifier, scaler) if args.use_test else 0.0       # on the testing set
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)

    # Save the trained network
    print('\nSaving trained network...')
    model_path = os.path.join(save_dir, f"FordA_{netw}_full{suffix}_model_{i}.pt")
    torch.save(model.state_dict(), model_path) # save reservoir random parameters (epsilon, gamma, h2h, x2h, bias)
    scaler_path = os.path.join(save_dir, f"FordA_{netw}_full{suffix}_scaler_{i}.pkl")
    joblib.dump(scaler, scaler_path) # save scaler
    classifier_path = os.path.join(save_dir, f"FordA_{netw}_full{suffix}_classifier_{i}.pkl")
    joblib.dump(classifier, classifier_path) # save classifier
    print()
    
# Save results
print('Saving results...')
f = open(os.path.join(save_dir, f"FordA_log_{netw}_full{suffix}.txt"), "a")

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
print('\nDone!')
