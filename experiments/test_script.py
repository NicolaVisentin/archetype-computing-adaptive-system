# Script to test "by hand" a certain architecture

import numpy as np
import torch
import joblib
from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from PIL import Image
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
model_dir = Path('results/trained_architectures') # folder with the trained architectures to test
imgs_dir = Path('acds/benchmarks/raw')            # folder with data

# =========================================================
# Re-create the full, saved network
# =========================================================

# Create an object of the reservoir
n_inp = 1
#n_hid = 256
n_hid = 800
dt = 0.042
gamma = (2.7 - 2.7 / 2.0, 2.7 + 2.7 / 2.0)
epsilon = (4.7 - 4.7 / 2.0, 4.7 + 4.7 / 2.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model_params = torch.load(model_dir/"sMNIST_RON_full_800hidden_model_0.pt", map_location=device)
model.load_state_dict(model_params)
model.eval()

# Load saved scaler and classifier
scaler = joblib.load(model_dir/"sMNIST_RON_full_800hidden_scaler_0.pkl")
classifier = joblib.load(model_dir/"sMNIST_RON_full_800hidden_classifier_0.pkl")


# =========================================================
# Load desired image
# =========================================================

# # Load an image from MNIST dataset
# transform = transforms.ToTensor()
# mnist_test_dataset = datasets.MNIST(
#     root=imgs_dir, 
#     train=False, 
#     transform=transform, 
#     download=False
# )                                      # load test dataset
# image_mnist, _ = mnist_test_dataset[90] # extract first image (1,28,28), grayscale, torch tensor, float32 values in [0,1]
# image_tensor = image_mnist.to(device)  # (1,28,28), grayscale, torch tensor, on proper device, float32 values in [0,1]
# image_test = image_tensor.view(1,-1,1) # resize to (1, 784, 1), as required by forward method of the model

# Load the custom image and convert it properly
transform = transforms.Compose([
    transforms.Grayscale(),            # convert to grayscale
    transforms.Resize((28, 28)),       # resize to (1, 28, 28)
    transforms.ToTensor(),             # convert to torch tensor float32 in [0, 1]
    transforms.Lambda(lambda x: 1 - x) # invert to have white digit on black background
])
image_path = imgs_dir/'test_MNIST_6.png'
image = Image.open(image_path)
image_tensor = transform(image).to(device) # (1,28,28), grayscale, torch tensor, on proper device, float32 values in [0,1]
image_test = image_tensor.view(1,-1,1)     # resize to (1, 784, 1), as required by forward method of the model


# =========================================================
# Test the network
# =========================================================

# Feed it to the model
activations = model(image_test)[-1][0] # out is (1, num_hidden): contains the last states of all hidden units: those are our activations
activations = activations.cpu()        # pass to cpu (if not already there)
activations = scaler.transform(activations) 
pred = classifier.predict(activations)[0]

# Show prediction
probs = classifier.predict_proba(activations).squeeze()

plt.figure()
plt.imshow(image_tensor.cpu().squeeze(), cmap='gray')
plt.title(f'Prediction: {pred}')
plt.figtext(0.5, 0.03, f'Classes probabilities: {np.round(100*probs, 1)} %', ha='center', va='center')
plt.show()