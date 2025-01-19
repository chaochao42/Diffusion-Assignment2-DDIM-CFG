
from dataset import TwoDimDataClass, get_data_iterator
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Output
from IPython.display import display, clear_output
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import io
from network import SimpleNet
from ddpm import BaseScheduler, DiffusionModule
from chamferdist import chamfer_distance

target_ds = TwoDimDataClass(dataset_type='swiss_roll', 
                            N=1000000, 
                            batch_size=256)

prior_ds = TwoDimDataClass(dataset_type='gaussian_centered',
                           N=1000000,
                           batch_size=256)

num_vis_particles = 500
sample_f = target_ds[0:num_vis_particles]
sample_b = prior_ds[0:num_vis_particles]

device = "cuda:0"
config = {
    "num_diffusion_steps": 1000,
    "dim_hids": [128, 128, 128],
    "lr": 1e-3,
    "batch_size": 128,
    "num_train_iters": 5000,
    "device": device,
}

###################

def build_ddpm(config):
    network = SimpleNet(dim_in=2, 
                        dim_out=2, 
                        dim_hids=config["dim_hids"], 
                        num_timesteps=config["num_diffusion_steps"]
                       )
    var_scheduler = BaseScheduler(config["num_diffusion_steps"])

    ddpm = DiffusionModule(network, var_scheduler).to(config["device"])
    return ddpm

ddpm = build_ddpm(config)

def figure2image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

# Initialize the model.
ddpm = build_ddpm(config)

pbar = tqdm(range(config["num_train_iters"]))
optimizer = torch.optim.Adam(ddpm.parameters(), lr=config["lr"])
train_dl = torch.utils.data.DataLoader(target_ds, batch_size=config["batch_size"])
train_iter = get_data_iterator(train_dl)

losses = []
images = []

for step in pbar:
    optimizer.zero_grad()
    
    batch_x = next(train_iter)
    batch_x = batch_x.to(device)
    loss = ddpm.compute_loss(batch_x)

    loss.backward()
    optimizer.step()
    pbar.set_description(f"loss: {loss.item():.4f}")
    
    losses.append(loss.item())
    
    if step % 4999 == 0:
        with torch.no_grad():
            ####
            # NOTE: If you haven't implemented the `ddim_p_sample_loop` method,
            # use the `p_sample_loop` method instead.
            x0 = ddpm.p_sample_loop(shape=(num_vis_particles, 2)).cpu()
            # x0 = ddpm.ddim_p_sample_loop(shape=(num_vis_particles, 2)).cpu()
            ####
            fig, ax = plt.subplots(1,1)
            ax.scatter(x0[:,0], x0[:,1])
            ax.set_title(f"Samples at {step}-iteration")
            clear_output(wait=True)
            plt.show()
            img = figure2image(fig)
            images.append(img)
                
if len(images) > 0:
    slider = IntSlider(min=0, max=len(images)-1, step=1, value=1)
    output = Output()
    def display_image(index):
        with output:
            output.clear_output(wait=True)
            display(images[index])
    interact(display_image, index=slider)
    display(output)
    plt.plot(losses)
    plt.title("Loss curve")

num_eval_particles = 2048
pc_ref = target_ds[:num_eval_particles]
pc_gen = ddpm.p_sample_loop(shape=(num_eval_particles, 2))

pc_gen = pc_gen.reshape(1, num_eval_particles, 2)
pc_ref = pc_ref.reshape(1, num_eval_particles, 2)
with torch.no_grad():
    cd = chamfer_distance(
            pc_gen.reshape(-1, 2).cpu().numpy(),
            pc_ref.reshape(-1, 2).cpu().numpy(),
        )
    print(f"DDPM Chamfer Distance: {cd.item():.4f}")

# Visualize samples with the target distribution.
pc_gen = pc_gen.reshape(num_eval_particles, 2).cpu().numpy()
pc_ref = pc_ref.reshape(num_eval_particles, 2).cpu().numpy()

fig, ax = plt.subplots(1,1)
ax.scatter(pc_ref[:,0], pc_ref[:,1], alpha=0.1, label="target distribution")
ax.scatter(pc_gen[:,0], pc_gen[:,1], alpha=0.1, label="samples")
ax.legend()
plt.show()