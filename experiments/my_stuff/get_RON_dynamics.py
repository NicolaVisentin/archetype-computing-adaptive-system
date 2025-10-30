# =========================================================
# Setup
# =========================================================

# Imports
import numpy as np
import torch
import joblib
from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from torchvision import transforms, datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pathlib import Path

# Choose device (CPU/GPU)
device = (torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Get relevant paths
curr_dir = Path(__file__).parent                           # current folder
model_dir = Path(curr_dir/'results/trained_architectures') # folder with the trained architectures
imgs_dir = Path('src/acds/benchmarks/raw')                 # folder with datasave_dataset_dir.mkdir(parents=True, exist_ok=True)
plots_dir = curr_dir/'plots'/Path(__file__).stem           # folder to save plots
plots_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# Re-create the full, saved network
# =========================================================

# Create an object of the reservoir
n_inp = 1
n_hid = 6
dt = 0.042
gamma = (2.7 - 1 / 2.0, 2.7 + 1 / 2.0)
epsilon = (0.51 - 0.5 / 2.0, 0.51 + 0.5 / 2.0)

model = RandomizedOscillatorsNetwork(
    n_inp=n_inp,
    n_hid=n_hid,
    dt=dt,
    gamma=gamma,
    epsilon=epsilon,
    diffusive_gamma=0.0,
    rho=9,
    input_scaling=1.0,
    topology='full',
    reservoir_scaler=1.0,
    sparsity=0.0,
    device=device,
).to(device)

# Load and assign saved parameters to the reservoir
model_params = torch.load(model_dir/"sMNIST_RON_full_6hidden/sMNIST_RON_full_6hidden_model_5.pt", map_location=device)
model.load_state_dict(model_params)
model.eval()

# # Load saved scaler and classifier
# scaler = joblib.load(model_dir/"sMNIST_RON_full_6hidden/sMNIST_RON_full_6hidden_scaler_5.pkl")
# classifier = joblib.load(model_dir/"sMNIST_RON_full_6hidden/sMNIST_RON_full_6hidden_classifier_5.pkl")


# =========================================================
# Load desired images
# =========================================================

# # Load an image from MNIST dataset
# transform = transforms.ToTensor()
# mnist_test_dataset = datasets.MNIST(
#     root=imgs_dir, 
#     train=False, 
#     transform=transform, 
#     download=False
# )                                      # load test dataset
# image_mnist, _ = mnist_test_dataset[0] # extract first image (1,28,28), grayscale, torch tensor, float32 values in [0,1]
# image_tensor = image_mnist.to(device)  # (1,28,28), grayscale, torch tensor, on proper device, float32 values in [0,1]
# image_test = image_tensor.view(1,-1,1) # resize to (1, 784, 1), as required by forward method of the model

# Custom image
image_test = torch.zeros((1, 784, 1), device=device) # completely black image (null input)


# =========================================================
# Get the dynamics of the reservoir
# =========================================================

# Feed it to the model
out = model(image_test)                   # tuple (states_hist, last_states)
states_histories = out[0]                 # hidden states time history (batch_size, num_steps, n_hid). In this case (1, 784, n_hid)                # last states (batch_size, n_hid)
states_histories = states_histories.cpu() # pass to cpu (if not already there)

# Show evolution of the states and their velocities
time = np.arange(0, dt*states_histories.shape[1], dt)
velocities_histories = velocity = np.diff(states_histories, axis=1) / dt

fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(n_hid):
    ax1.plot(time, states_histories[0,:,i], label=f'y{i+1}(t)')
    ax2.plot(time[:-1], velocities_histories[0,:,i], label=f'yd{i+1}(t)')
ax1.grid(True)
ax2.grid(True)
ax1.set_xlabel('t [s]')
ax2.set_xlabel('t [s]')
ax1.set_ylabel('y')
ax2.set_ylabel('yd')
ax1.set_title('Hidden states positions')
ax2.set_title('Hidden states velocities')
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig(plots_dir/'states_evolution', bbox_inches='tight')
#plt.show()

# Show (y, yd) in y,yd plane
fig, axs = plt.subplots(3,2, figsize=(12,9))
for i, ax in enumerate(axs.flatten()):
    sc = ax.scatter(states_histories[0,:-1,i], velocities_histories[0,:,i], c=time[:-1], cmap='viridis', label='t=0')
    ax.grid(True)
    ax.set_xlabel('y')
    ax.set_ylabel('yd')
    ax.set_title(f'hidden state {i+1}')
    ax.legend()
plt.tight_layout()
plt.savefig(plots_dir/'state_space', bbox_inches='tight')
plt.show()