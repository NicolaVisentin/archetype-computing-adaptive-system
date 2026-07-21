# =========================================================
# Setup
# =========================================================

# Imports
import numpy as np
import torch
import joblib
from tqdm import tqdm
from acds.archetypes import RandomizedOscillatorsNetwork
from acds.benchmarks import get_adiac_data
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Plots settings
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'cm',
})

# Choose device (CPU/GPU)
device = (torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Get relevant paths
curr_dir = Path(__file__).parent # current folder
model_dir = Path(curr_dir.parent/'trained_architectures') # folder with the trained architectures to test
dataset_dir = Path('src/acds/benchmarks/raw/adiac') # folder with datasets
plots_dir = curr_dir.parent/'plots'/curr_dir.stem/Path(__file__).stem # folder to save plots
save_results_dir = Path(curr_dir.parent/'results'/curr_dir.stem/Path(__file__).stem) # folder to save data


# =========================================================
# Script settings
# =========================================================

n_hid = 6 # dimension of the hidden state (number of oscillators)
architecture_to_test = 'Adiac_RON_full_6hidden' # path of the folder containing trained scaler, model and classifier
dt = 0.02 # dt of the RON reservoir
rho = 0.99 # spectral radius of the hidden-to-hidden weight matrix
# !!! remember to choose the best model when loading the model, scaler and classifier !!!

# ------------

plots_dir = plots_dir/architecture_to_test
save_results_dir = save_results_dir/architecture_to_test
plots_dir.mkdir(parents=True, exist_ok=True)
save_results_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# Re-create the full, saved network
# =========================================================

# Create an object of the reservoir
n_inp = 1
gamma = (2.7 - 1 / 2.0, 2.7 + 1 / 2.0) # dummy
epsilon = (0.51 - 0.5 / 2.0, 0.51 + 0.5 / 2.0) # dummy

model = RandomizedOscillatorsNetwork(
    n_inp=n_inp,
    n_hid=n_hid,
    dt=dt,
    gamma=gamma, # dummy
    epsilon=epsilon, # dummy
    rho=rho, # dummy
    input_scaling=10.0, # dummy
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

    return classifier.score(activations, ys), output_batch, activations

_, _, test_loader = get_adiac_data(
    root_path = dataset_dir,
    bs_train = 30,
    bs_test = 30,
    whole_train=True,
)

# Test on test set
score, last_states, activations = test(test_loader, classifier, scaler)
last_states = last_states.cpu()
print(f'Accuracy on the test set: {score}')

# Visualize the activations for all the test set
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for i in range(last_states.shape[1]):
    ax1.scatter(last_states[:,i], (i+1)*np.ones(len(last_states)), label=f'Component {i+1}')
ax1.set_title('Last states')
ax1.set_xlabel(r'$y(t_{f})$')
ax1.set_ylabel('component')
ax1.grid(True)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

for i in range(activations.shape[1]):
    ax2.scatter(activations[:,i], (i+1)*np.ones(len(activations)), label=f'Component {i+1}')
ax2.set_title('Activations')
ax2.set_xlabel(r'$\tilde{y}$')
ax2.set_ylabel('component')
ax2.grid(True)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(plots_dir/'all_testset_activations', bbox_inches='tight')
#plt.show()


# =========================================================
# Test on one single sequence
# =========================================================   

# Extract first sequence from the dataset 
x_batch, y_batch = next(iter(test_loader))
input_sequence, label = x_batch[:1], y_batch[:1] # shape (1, L, 1) and (1, 1)

# Feed it to the model
out = model(input_sequence.to(device)) # list (states_hist, last_states)

states_histories = out[0] # hidden states time history (batch_size, num_steps, n_hid). In this case (1, 176, n_hid)
states_histories = states_histories.cpu() # pass to cpu (if not already there)

last_states = out[-1][0] # last hidden states (batch_size, n_hid). Contains the last states of all hidden units
last_states = last_states.cpu() # pass to cpu (if not already there)
activations = scaler.transform(last_states) # these are our "real" activations (also apply scaling to the output)
pred = classifier.predict(activations)[0] # prediction with the trained classifier

# Show prediction and probabilities
probs = classifier.predict_proba(activations).squeeze()

plt.figure(figsize=(8, 4.5))
plt.bar(np.arange(37)+1, probs, color='skyblue')
plt.title(f'Prediction: {int(pred+1)} | label: {int(label+1)}\n(model test set accuracy: {score:.4f})', fontsize=14)
plt.xlabel(r'classes', fontsize=14)
plt.ylabel(r'probability', fontsize=14)
plt.xticks(np.arange(37)+1) 

plt.tight_layout()
plt.savefig(plots_dir/'example_prediction', bbox_inches='tight')
#plt.show()

# Show dynamics of the reservoir: states, velocities, accelerations and input in time (!! MAX FIRST 15 STATES !!)
if n_hid > 15:
    n_hid_show = 15
else:
    n_hid_show = n_hid
time = np.arange(0, dt*states_histories.shape[1], dt)
velocities_histories = np.diff(states_histories, axis=1) / dt
accelerations_histories = np.diff(velocities_histories, axis=1) / dt
input_history = input_sequence.cpu().numpy()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,12))
for i in range(n_hid_show):
    ax1.plot(time, states_histories[0,:,i], label=rf'$y_{{{i+1}}}(t)$')
    ax2.plot(time[:-1], velocities_histories[0,:,i], label=rf'$\dot{{y}}_{{{i+1}}}(t)$')
    ax3.plot(time[:-2], accelerations_histories[0,:,i], label=rf'$\ddot{{y}}_{{{i+1}}}(t)$')
ax4.plot(time, input_history[0,:,0])
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True)
    ax.set_xlabel('t [s]')
ax1.set_ylabel(r'$y$')
ax2.set_ylabel(r'$\dot{y}$')
ax3.set_ylabel(r'$\ddot{y}$')
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
plt.show()

# Show dynamics of the reservoir: (y, yd) in y,yd plane
n_cols = min(3, n_hid_show)
n_rows = int(np.ceil(n_hid_show / n_cols))

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9))
if n_hid_show == 1:
    axs = np.array([axs])
else:
    axs = axs.flatten()

for i in range(n_hid_show):
    sc = axs[i].scatter(states_histories[0, :-1, i], velocities_histories[0, :, i], 
                        c=time[:-1], cmap='viridis', s=10, alpha=0.6, label='t=0')
    axs[i].grid(True)
    axs[i].set_xlabel(r'$y$')
    axs[i].set_ylabel(r'$\dot{y}$')
    axs[i].set_title(f'Hidden state {i+1}')
    axs[i].legend(loc='upper left')

for i in range(n_hid, len(axs)):
    axs[i].set_visible(False)

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
# Compare with another input
# =========================================================

# Load another sequence from the dataset
input_sequence_2, label_2 = x_batch[1:2], y_batch[1:2] # shape (1, L, 1) and (1, 1)

# Feed input to the model
out2 = model(input_sequence_2.to(device)) # tuple (states_hist, last_states)
states_histories2 = out2[0] # hidden states time history (batch_size, num_steps, n_hid). In this case (1, 784, n_hid)
states_histories2 = states_histories2.cpu() # pass to cpu (if not already there)

# Compares states and inputs in time
time2 = np.arange(0, dt*states_histories2.shape[1], dt)
input_history2 = input_sequence_2.cpu().numpy()

fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 9))
if n_hid_show == 1:
    axs = np.array([axs])
else:
    axs = axs.flatten()

for i in range(n_hid_show):
    axs[i].plot(time, states_histories[0, :, i], label='test1', alpha=0.8)
    axs[i].plot(time2, states_histories2[0, :, i], label='test2', alpha=0.8)
    axs[i].grid(True)
    axs[i].set_xlabel('t [s]')
    axs[i].set_ylabel('y')
    axs[i].set_title(f'Component {i+1}')
    axs[i].legend(loc='upper left')

for i in range(n_hid_show, len(axs)):
    axs[i].set_visible(False)

plt.tight_layout()
plt.savefig(plots_dir/'states_comparison', bbox_inches='tight')
plt.show()
