# Script to get the hidden dynamics of the reservoir

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
imgs_dir = Path('src/acds/benchmarks/raw')                 # folder with data


# =========================================================
# Re-create the full, saved network
# =========================================================

# Create an object of the reservoir
n_inp = 1
n_hid = 2
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
model_params = torch.load(model_dir/"sMNIST_RON_full_2hidden/sMNIST_RON_full_2hidden_model_0.pt", map_location=device)
model.load_state_dict(model_params)
model.eval()

# Load saved scaler and classifier
scaler = joblib.load(model_dir/"sMNIST_RON_full_2hidden/sMNIST_RON_full_2hidden_scaler_0.pkl")
#classifier = joblib.load(model_dir/"sMNIST_RON_full_2hidden/sMNIST_RON_full_2hidden_classifier_0.pkl")


# =========================================================
# Load desired images
# =========================================================

# Load an image from MNIST dataset
transform = transforms.ToTensor()
mnist_test_dataset = datasets.MNIST(
    root=imgs_dir, 
    train=False, 
    transform=transform, 
    download=False
)                                      # load test dataset
image_mnist, _ = mnist_test_dataset[0] # extract first image (1,28,28), grayscale, torch tensor, float32 values in [0,1]
image_tensor = image_mnist.to(device)  # (1,28,28), grayscale, torch tensor, on proper device, float32 values in [0,1]
image_test = image_tensor.view(1,-1,1) # resize to (1, 784, 1), as required by forward method of the model

# Custom image
#image_test = torch.zeros((1, 784, 1), device=device) # completely black image (null input)

# =========================================================
# Get the dynamics of the reservoir
# =========================================================

# Feed it to the model
activations = model(image_test)[0] # hidden states time history (batch_size, num_steps, n_hid). In this case (1,784,2)
activations = activations.cpu()    # pass to cpu (if not already there)
#activations = scaler.transform(activations) 

# Show evolution of the states
time = np.arange(0, dt*activations.shape[1], dt)

plt.figure()
for i in range(n_hid):
    plt.plot(time, activations[0,:,i], label=f'y{i}(t)')
plt.grid(True)
plt.xlabel('t [s]')
plt.ylabel('y')
plt.title('Hidden states')
plt.legend()

plt.tight_layout()
plt.show()