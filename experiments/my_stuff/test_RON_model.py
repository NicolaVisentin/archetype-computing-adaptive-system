# Script to test "by hand" a certain architecture

# =========================================================
# Setup
# =========================================================

# Imports
import numpy as np
import torch
import joblib
from tqdm import tqdm
from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks.mnist import get_mnist_data
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
curr_dir = Path(__file__).parent                           # current folder
model_dir = Path(curr_dir/'results/trained_architectures') # folder with the trained architectures to test
imgs_dir = Path('src/acds/benchmarks/raw')                 # folder with datasets


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

# Load and assign saved parameters to the reservoir (! this assignes only epsilon, gamma, h2h, x2h, bias. Other
# parameters must be initialized correctly !)
model_params = torch.load(model_dir/"sMNIST_RON_full_6hidden/sMNIST_RON_full_6hidden_model_5.pt", map_location=device)
model.load_state_dict(model_params)
model.eval()

# Load saved scaler and classifier
scaler = joblib.load(model_dir/"sMNIST_RON_full_6hidden/sMNIST_RON_full_6hidden_scaler_5.pkl")
classifier = joblib.load(model_dir/"sMNIST_RON_full_6hidden/sMNIST_RON_full_6hidden_classifier_5.pkl")


# =========================================================
# Test on the test set
# =========================================================

# Function to test the trained classifier
@torch.no_grad()
def test(data_loader, classifier, scaler):
    last_states, ys = [], []
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
        last_states.append(output.cpu())
        ys.append(labels)

    last_states = torch.cat(last_states, dim=0).numpy() # shape (num_train_images, num_hidden_units)
    activations = scaler.transform(last_states)        
    ys = torch.cat(ys, dim=0).numpy()                   # shape (num_train_images,)

    return classifier.score(activations, ys), last_states, activations

_, _, test_loader = get_mnist_data(
    root=imgs_dir, 
    bs_train=6000, 
    bs_test=6000,
    #valid_perc=0
)

# Test on test set
score, last_states, activations = test(test_loader, classifier, scaler)
print(score)

# Visualize the activations for all the test set
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
for i in range(last_states.shape[1]):
    ax1.scatter(last_states[:,i], (i+1)*np.ones(len(last_states)), label=f'Component {i+1}')
ax1.set_title('Last states')
ax1.set_xlabel(r'$y(t_{f})$')
ax1.set_ylabel('component')
ax1.grid(True)

for i in range(activations.shape[1]):
    ax2.scatter(activations[:,i], (i+1)*np.ones(len(activations)), label=f'Component {i+1}')
ax2.set_title('Activations')
ax2.set_xlabel(r'$\tilde{y}$')
ax2.set_ylabel('component')
ax2.grid(True)

plt.tight_layout()
plt.show()


# =========================================================
# Load desired image
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

# # Load the custom image and convert it properly
# transform = transforms.Compose([
#     transforms.Grayscale(),            # convert to grayscale
#     transforms.Resize((28, 28)),       # resize to (1, 28, 28)
#     transforms.ToTensor(),             # convert to torch tensor float32 in [0, 1]
#     transforms.Lambda(lambda x: 1 - x) # invert to have white digit on black background
# ])
# image_path = imgs_dir/'test_MNIST_2.png'
# image = Image.open(image_path)
# image_tensor = transform(image).to(device) # (1,28,28), grayscale, torch tensor, on proper device, float32 values in [0,1]
# image_test = image_tensor.view(1,-1,1)     # resize to (1, 784, 1), as required by forward method of the model


# =========================================================
# Test the network
# =========================================================

# Feed it to the model
out = model(image_test)                     # list (states_hist, last_states)
last_states = out[-1][0]                    # last hidden states (batch_size, n_hid). Contains the last states of all hidden units
last_states = last_states.cpu()             # pass to cpu (if not already there)
activations = scaler.transform(last_states) # these are our "real" activations (also apply scaling to the output)
pred = classifier.predict(activations)[0]   # prediction with the trained classifier

# Show prediction
probs = classifier.predict_proba(activations).squeeze()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(image_tensor.cpu().squeeze(), cmap='gray')
ax1.set_title('Input')

ax2.bar(np.arange(10), probs, color='skyblue')
ax2.set_title(f'Prediction: {pred}')
ax2.set_xlabel('classes')
ax2.set_ylabel('probability')
ax2.set_xticks(np.arange(10)) 

plt.tight_layout()
plt.show()