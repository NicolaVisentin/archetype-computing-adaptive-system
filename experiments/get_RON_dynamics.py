# Script to get the hidden dynamics of the reservoir

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
model_dir = Path('results/trained_architectures') # folder with the trained architectures
imgs_dir = Path('acds/benchmarks/raw')            # folder with data

# =========================================================
# Re-create the full, saved network
# =========================================================

# Create an object of the reservoir
n_inp = 1
n_hid = 2 # default 256
dt = 0.042
gamma = (2.7 - 2.7 / 2.0, 2.7 + 2.7 / 2.0)
epsilon = (4.7 - 4.7 / 2.0, 4.7 + 4.7 / 2.0)

model = RandomizedOscillatorsNetwork(
    n_inp=n_inp,
    n_hid=n_hid,
    dt=dt,
    gamma=gamma,
    epsilon=epsilon,
    diffusive_gamma=0.0,
    rho=0.99,
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


# =========================================================
# Get the dynamics of the reservoir
# =========================================================

# Feed it to the model
activations = model(image_test)[0] # hidden states time history (batch_size, num_steps, n_hid). In this case (1,784,2)
activations = activations.cpu()    # pass to cpu (if not already there)
#activations = scaler.transform(activations) 

# Show evolution of the states
time = np.arange(0, dt*activations.shape[1], dt)

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(time, activations[0,:,0], label=r'$y_1(t)$')
ax1.grid(True)
ax1.set_xlabel('t [s]')
ax1.set_ylabel(r'$y_1$')
ax1.set_title('First hidden state')

ax2.plot(time, activations[0,:,1], label=r'$y_2(t)$')
ax2.grid(True)
ax2.set_xlabel('t [s]')
ax2.set_ylabel(r'$y_2$')
ax2.set_title('Second hidden state')

plt.tight_layout()
plt.show()