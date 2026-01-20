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
curr_dir = Path(__file__).parent                                # current folder
model_dir = Path(curr_dir/'trained_architectures')              # folder with the trained architectures to test
imgs_dir = Path('src/acds/benchmarks/raw')                      # folder with datasets
plots_dir = curr_dir/'plots'/Path(__file__).stem                # folder to save plots
save_results_dir = Path(curr_dir/'results'/Path(__file__).stem) # folder to save data


# =========================================================
# Script settings
# =========================================================

n_hid = 6 # dimension of the hidden state (number of oscillators)
architecture_to_test = 'sMNIST_RON_full_6hidden' # path of the folder containing trained scaler, model and classifier
image_to_test = 0 # if it is an integer i, loads the i-th image from MNIST test set. Otherwise 'black' or 'custom'

plots_dir = plots_dir/architecture_to_test
save_results_dir = save_results_dir/architecture_to_test
plots_dir.mkdir(parents=True, exist_ok=True)
save_results_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# Re-create the full, saved network
# =========================================================

# Create an object of the reservoir
n_inp = 1
dt = 0.042 # NOT dummy
gamma = (2.7 - 1 / 2.0, 2.7 + 1 / 2.0) # dummy
epsilon = (0.51 - 0.5 / 2.0, 0.51 + 0.5 / 2.0) # dummy

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
model_params = torch.load(model_dir/architecture_to_test/f"{architecture_to_test}_model_5.pt", map_location=device)
model.load_state_dict(model_params)
model.eval()

# Load saved scaler and classifier
scaler = joblib.load(model_dir/architecture_to_test/f"{architecture_to_test}_scaler_5.pkl")
classifier = joblib.load(model_dir/architecture_to_test/f"{architecture_to_test}_classifier_5.pkl")


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
    valid_perc=0 # with valid_perc=0 loads empty validation loader and "full" test loader
)

# Test on test set
score, last_states, activations = test(test_loader, classifier, scaler)
print(f'Accuracy on the test set: {score}')

# Visualize the activations for all the test set
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
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
plt.savefig(plots_dir/'all_testset_activations', bbox_inches='tight')
plt.show()


# =========================================================
# Test on one single image
# =========================================================

# Load image to test
if image_to_test == 'black':
    image_test = torch.zeros((1, 784, 1), device=device) # completely black image (null input)
elif image_to_test == 'custom':
    transform = transforms.Compose([
        transforms.Grayscale(),            # convert to grayscale
        transforms.Resize((28, 28)),       # resize to (1, 28, 28)
        transforms.ToTensor(),             # convert to torch tensor float32 in [0, 1]
        transforms.Lambda(lambda x: 1 - x) # invert to have white digit on black background
    ])
    image_path = imgs_dir/'test_MNIST_2.png'
    image = Image.open(image_path)
    image_tensor = transform(image).to(device) # (1,28,28), grayscale, torch tensor, on proper device, float32 values in [0,1]
    image_test = image_tensor.view(1,-1,1)     # resize to (1, 784, 1), as required by forward method of the model
else:
    transform = transforms.ToTensor()
    mnist_test_dataset = datasets.MNIST(
        root=imgs_dir, 
        train=False, 
        transform=transform, 
        download=False
    )                                                  # load test dataset
    image_mnist, _ = mnist_test_dataset[image_to_test] # extract desired image (1,28,28), grayscale, torch tensor, float32 values in [0,1]
    image_tensor = image_mnist.to(device)              # (1,28,28), grayscale, torch tensor, on proper device, float32 values in [0,1]
    image_test = image_tensor.view(1,-1,1)             # resize to (1, 784, 1), as required by forward method of the model

# Feed it to the model
out = model(image_test)                     # list (states_hist, last_states)

states_histories = out[0]                   # hidden states time history (batch_size, num_steps, n_hid). In this case (1, 784, n_hid)
states_histories = states_histories.cpu()   # pass to cpu (if not already there)

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
ax2.set_title(f'Prediction: {pred}\n(model test set accuracy: {score})')
ax2.set_xlabel('classes')
ax2.set_ylabel('probability')
ax2.set_xticks(np.arange(10)) 

plt.tight_layout()
plt.savefig(plots_dir/'example_prediction', bbox_inches='tight')
plt.show()

# Show dynamics of the reservoir: states, velocities, accelerations and input in time
time = np.arange(0, dt*states_histories.shape[1], dt)
velocities_histories = np.diff(states_histories, axis=1) / dt
accelerations_histories = np.diff(velocities_histories, axis=1) / dt
input_history = image_test.cpu().numpy()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,12))
for i in range(n_hid):
    ax1.plot(time, states_histories[0,:,i], label=f'y{i+1}(t)')
    ax2.plot(time[:-1], velocities_histories[0,:,i], label=f'yd{i+1}(t)')
    ax3.plot(time[:-2], accelerations_histories[0,:,i], label=f'ydd{i+1}(t)')
ax4.plot(time, input_history[0,:,0])
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
ax1.set_xlabel('t [s]')
ax2.set_xlabel('t [s]')
ax3.set_xlabel('t [s]')
ax4.set_xlabel('t [s]')
ax1.set_ylabel('y')
ax2.set_ylabel('yd')
ax3.set_ylabel('ydd')
ax4.set_ylabel('u')
ax1.set_title('Hidden states positions')
ax2.set_title('Hidden states velocities')
ax3.set_title('Hidden states accelerations')
ax4.set_title('Input')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig(plots_dir/'states_evolution', bbox_inches='tight')
#plt.show()

# Show dynamics of the reservoir: (y, yd) in y,yd plane
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

# Save dynamics of the reservoir
np.savez(
    save_results_dir/'RON_evolution.npz', 
    time = time[:-2],
    y = states_histories[0,:-2], 
    yd = velocities_histories[0,:-1], 
    ydd = accelerations_histories[0,:],
    u = input_history[0,:]
)


# =========================================================
# Compare with another image
# =========================================================

# Load another image from MNIST dataset
image2_mnist, _ = mnist_test_dataset[1] # extract second image (1,28,28), grayscale, torch tensor, float32 values in [0,1]. It is a 2
image2_tensor = image2_mnist.to(device)  # (1,28,28), grayscale, torch tensor, on proper device, float32 values in [0,1]
image2_test = image2_tensor.view(1,-1,1) # resize to (1, 784, 1), as required by forward method of the model

# Feed input to the model
out2 = model(image2_test)                   # tuple (states_hist, last_states)
states_histories2 = out2[0]                 # hidden states time history (batch_size, num_steps, n_hid). In this case (1, 784, n_hid)
states_histories2 = states_histories2.cpu() # pass to cpu (if not already there)

# Compares states and inputs in time
time2 = np.arange(0, dt*states_histories2.shape[1], dt)
input_history2 = image2_test.cpu().numpy()

fig, axs = plt.subplots(3, 2, figsize=(16,9))
for i, ax in enumerate(axs.flatten()):
    ax.plot(time, states_histories[0,:,i], label='image1')
    ax.plot(time2, states_histories2[0,:,i], label='image2')
    ax.grid(True)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('y')
    ax.set_title(f'Component {i+1}')
    ax.legend()
plt.tight_layout()
plt.savefig(plots_dir/'states_comparison', bbox_inches='tight')
plt.show()
