import torch
from torch import nn
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch.nn.functional as F

######################################################################
# PATCHED: Replace old LoRA attention processor imports with PEFT-based LoRA.
#
# OLD (diffusers < 0.22):
#   from diffusers.models.attention_processor import (
#       AttnAddedKVProcessor, AttnAddedKVProcessor2_0,
#       LoRAAttnAddedKVProcessor, LoRAAttnProcessor,
#       SlicedAttnAddedKVProcessor,
#   )
#   from diffusers.loaders import AttnProcsLayers
#
# NEW: We use peft's LoraConfig to inject LoRA layers into the UNet.
# This is the modern approach used by diffusers >= 0.25.
######################################################################
from peft import LoraConfig, get_peft_model


def get_t_schedule(num_train_timesteps, args, loss_weight=None):
    # Create a list of time steps from 0 to num_train_timesteps
    ts = list(range(num_train_timesteps))
    # set ts to U[0.02,0.98] as least
    assert (args.t_start >= 20) and (args.t_end <= 980)
    ts = ts[args.t_start:args.t_end]

    # If the scheduling strategy is 'random', choose args.num_steps random time steps without replacement
    if args.t_schedule == 'random':
        chosen_ts = np.random.choice(ts, args.num_steps, replace=True)

    # If the scheduling strategy is 'random_down', first exclude the first 30 and last 10 time steps
    # then choose a random time step from an interval that shrinks as step increases
    elif 'random_down' in args.t_schedule:
        interval_ratio = int(args.t_schedule[11:]) if len(args.t_schedule[11:]) > 0 else 5
        interval_ratio *= 0.1 
        chosen_ts = [np.random.choice(
                        ts[max(0,int((args.num_steps-step-interval_ratio*args.num_steps)/args.num_steps*len(ts))):\
                           min(len(ts),int((args.num_steps-step+interval_ratio*args.num_steps)/args.num_steps*len(ts)))], 
                     1, replace=True).astype(int)[0] \
                     for step in range(args.num_steps)]

    # If the scheduling strategy is 'fixed', parse the fixed time step from the string and repeat it args.num_steps times
    elif 'fixed' in args.t_schedule:
        fixed_t = int(args.t_schedule[5:])
        chosen_ts = [fixed_t for _ in range(args.num_steps)]

    # If the scheduling strategy is 'descend', parse the start time step from the string (or default to 1000)
    # then create a list of descending time steps from the start to 0, with length args.num_steps
    elif 'descend' in args.t_schedule:
        if 'quad' in args.t_schedule:   # no significant improvement
            descend_from = int(args.t_schedule[12:]) if len(args.t_schedule[7:]) > 0 else len(ts)
            chosen_ts = np.square(np.linspace(descend_from**0.5, 1, args.num_steps))
            chosen_ts = chosen_ts.astype(int).tolist()
        else:
            descend_from = int(args.t_schedule[7:]) if len(args.t_schedule[7:]) > 0 else len(ts)
            chosen_ts = np.linspace(descend_from-1, 1, args.num_steps, endpoint=True)
            chosen_ts = chosen_ts.astype(int).tolist()

    # If the scheduling strategy is 't_stages', the total number of time steps are divided into several stages.
    elif 't_stages' in args.t_schedule:
        num_stages = int(args.t_schedule[8:]) if len(args.t_schedule[8:]) > 0 else 2
        chosen_ts = []
        for i in range(num_stages):
            portion = ts[:int((num_stages-i)*len(ts)//num_stages)]
            selected_ts = np.random.choice(portion, args.num_steps//num_stages, replace=True).tolist()
            chosen_ts += selected_ts
    
    elif 'dreamtime' in args.t_schedule:
        assert 'dreamtime' in args.loss_weight_type
        loss_weight_sum = np.sum(loss_weight)
        p = [wt / loss_weight_sum for wt in loss_weight]
        N = args.num_steps
        def t_i(t, i, p):
            t = int(max(0, min(len(p)-1, t)))
            return abs(sum(p[t:]) - i/N)
        chosen_ts = []
        for i in range(N):
            t0 = 999
            selected_t = minimize(t_i, t0, args=(i, p), method='Nelder-Mead')
            selected_t = max(0, int(selected_t.x))
            chosen_ts.append(selected_t)
    else:
        raise ValueError(f"Unknown scheduling strategy: {args.t_schedule}")

    return chosen_ts


def get_loss_weights(betas, args):
    num_train_timesteps = len(betas)
    betas = torch.tensor(betas) if not torch.is_tensor(betas) else betas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)
    sigma_ks = []
    SNRs = []
    rhos = []
    m1 = 800
    m2 = 500
    s1 = 300
    s2 = 100
    for i in range(num_train_timesteps):
        sigma_ks.append(reduced_alpha_cumprod[i])
        SNRs.append(1 / reduced_alpha_cumprod[i])
        if args.loss_weight_type == 'rhos':
            rhos.append(1. * (args.sigma_y**2)/(sigma_ks[i]**2))
    def loss_weight(t):
        if args.loss_weight_type == None or args.loss_weight_type == 'none':
            return 1
        elif 'SNR' in args.loss_weight_type:
            if args.loss_weight_type == 'SNR':
                return 1 / SNRs[t]
            elif args.loss_weight_type == 'SNR_sqrt':
                return torch.sqrt(1 / SNRs[t])
            elif args.loss_weight_type == 'SNR_square':
                return (1 / SNRs[t])**2
            elif args.loss_weight_type == 'SNR_log1p':
                return torch.log(1 + 1 / SNRs[t])
        elif args.loss_weight_type == 'rhos':
            return 1 / rhos[t]
        elif 'alpha' in args.loss_weight_type:
            if args.loss_weight_type == 'sqrt_alphas_cumprod':
                return sqrt_alphas_cumprod[t]
            elif args.loss_weight_type == '1m_alphas_cumprod':
                return sqrt_1m_alphas_cumprod[t]**2
            elif args.loss_weight_type == 'alphas_cumprod':
                return alphas_cumprod[t]
            elif args.loss_weight_type == 'sqrt_alphas_1m_alphas_cumprod':
                return sqrt_alphas_cumprod[t] * sqrt_1m_alphas_cumprod[t]
        elif 'dreamtime' in args.loss_weight_type:
            if t > m1:
                return np.exp(-(t - m1)**2 / (2 * s1**2))
            elif t >= m2:
                return 1
            else:
                return np.exp(-(t - m2)**2 / (2 * s2**2))
        elif 'BAOAB' in args.loss_weight_type:
            return 2 * sqrt_1m_alphas_cumprod[t] ** 2 / alphas_cumprod[t]
        else:
            raise NotImplementedError
    weights = []
    for i in range(num_train_timesteps):
        weights.append(loss_weight(i))
    return weights


def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}, scheduler=None, lora_v=False, half_inference=False):
    batch_size = noisy_latents.shape[0]
    latent_model_input = torch.cat([noisy_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    if lora_v:
        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=noisy_latents.device, dtype=noisy_latents.dtype
        )
        alpha_t = alphas_cumprod[t] ** 0.5
        sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    # Convert inputs to half precision
    if half_inference:
        noisy_latents = noisy_latents.clone().half()
        text_embeddings = text_embeddings.clone().half()
        latent_model_input = latent_model_input.clone().half()
    if guidance_scale == 1.:
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v:
            noise_pred = noisy_latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
    else:
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v:
            noise_pred = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    noise_pred = noise_pred.float()
    return noise_pred


def predict_noise0_diffuser_multistep(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}, scheduler=None, steps=1, eta=0, half_inference=False):
    latents = noisy_latents
    t_start = t.item()
    if not (0 < t_start <= 1000):
        raise ValueError(f"t must be between 0 and 1000, get {t_start}")
    if t_start > steps:
        step_size = t_start // steps
        indices = [int((steps - i) * step_size) for i in range(steps)]
        if indices[0] != t_start:
            indices[0] = t_start
    else:
        indices = [int((t_start - i)) for i in range(t_start)]
    if indices[-1] != 0:
        indices.append(0)
    for i in range(len(indices)):
        t = torch.tensor([indices[i]] * t.shape[0], device=t.device)
        noise_pred = predict_noise0_diffuser(unet, latents, text_embeddings, t, guidance_scale=guidance_scale, \
                                             cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, half_inference=half_inference)
        pred_latents = scheduler.step(noise_pred, t, latents).pred_original_sample
        if indices[i+1] == 0:
            alpha_bar_t_start = scheduler.alphas_cumprod[indices[0]].clone().detach()
            return (noisy_latents - torch.sqrt(alpha_bar_t_start)*pred_latents) / (torch.sqrt(1 - alpha_bar_t_start))
        alpha_bar = scheduler.alphas_cumprod[indices[i]].clone().detach()
        alpha_bar_prev = scheduler.alphas_cumprod[indices[i+1]].clone().detach()
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(latents)
        mean_pred = (
            pred_latents * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(latents.shape) - 1)))
        )
        latents = mean_pred + nonzero_mask * sigma * noise


def sds_vsd_grad_diffuser(unet, noisy_latents, noise, text_embeddings, t, unet_phi=None, guidance_scale=7.5, \
                        grad_scale=1, cfg_phi=1., generation_mode='sds', phi_model='lora', \
                            cross_attention_kwargs={}, multisteps=1, scheduler=None, lora_v=False, \
                                half_inference = False):
    ######################################################################
    # PATCHED: For PEFT-based LoRA, we disable LoRA by calling
    # unet_phi.disable_adapter_layers() on the shared unet instead of
    # passing cross_attention_kwargs={'scale': 0}.
    # After inference, we re-enable with unet_phi.enable_adapter_layers().
    ######################################################################
    if generation_mode == 'vsd' and phi_model == 'lora' and not lora_v:
        # Disable LoRA so the pretrained UNet runs without LoRA influence
        unet.disable_adapter_layers()

    with torch.no_grad():
        if multisteps > 1:
            noise_pred = predict_noise0_diffuser_multistep(unet, noisy_latents, text_embeddings, t, guidance_scale=guidance_scale, cross_attention_kwargs={}, scheduler=scheduler, steps=multisteps, eta=0., half_inference=half_inference)
        else:
            noise_pred = predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, guidance_scale=guidance_scale, cross_attention_kwargs={}, scheduler=scheduler, half_inference=half_inference)

    if generation_mode == 'vsd' and phi_model == 'lora' and not lora_v:
        # Re-enable LoRA for phi model inference
        unet.enable_adapter_layers()

    if generation_mode == 'sds':
        grad = grad_scale * (noise_pred - noise)
        noise_pred_phi = noise
    elif generation_mode == 'vsd':
        with torch.no_grad():
            # For PEFT LoRA: unet_phi IS unet with LoRA enabled, so just call it normally
            # No cross_attention_kwargs needed for scale control
            noise_pred_phi = predict_noise0_diffuser(unet_phi, noisy_latents, text_embeddings, t, guidance_scale=cfg_phi, cross_attention_kwargs={}, scheduler=scheduler, lora_v=lora_v, half_inference=half_inference)
        grad = grad_scale * (noise_pred - noise_pred_phi.detach())

    grad = torch.nan_to_num(grad)
    return grad, noise_pred.detach().clone(), noise_pred_phi.detach().clone()


def phi_vsd_grad_diffuser(unet_phi, latents, noise, text_embeddings, t, cfg_phi=1., grad_scale=1, cross_attention_kwargs={}, scheduler=None, lora_v=False, half_inference=False):
    loss_fn = nn.MSELoss()
    clean_latents = scheduler.step(noise, t, latents).pred_original_sample
    # PATCHED: no cross_attention_kwargs needed for PEFT LoRA
    noise_pred = predict_noise0_diffuser(unet_phi, latents, text_embeddings, t, guidance_scale=cfg_phi, cross_attention_kwargs={}, scheduler=scheduler, half_inference=half_inference)
    if lora_v:
        target = scheduler.get_velocity(clean_latents.detach(), noise, t)
    else:
        target = noise
    loss = loss_fn(noise_pred, target)
    loss *= grad_scale
    return loss


def extract_lora_diffusers(unet, device):
    ######################################################################
    # PATCHED: Complete rewrite using PEFT-based LoRA injection.
    #
    # OLD approach: Manually created LoRAAttnProcessor for each attention
    #   layer and set them via unet.set_attn_processor().
    #
    # NEW approach: Use peft's LoraConfig to inject LoRA adapters into
    #   the UNet's attention layers. This is the standard modern approach.
    #   The UNet itself becomes the "unet_phi" — with LoRA enabled it acts
    #   as epsilon_phi, with LoRA disabled it acts as epsilon_pretrain.
    ######################################################################
    # Freeze all UNet parameters first
    unet.requires_grad_(False)

    # Configure LoRA for the cross-attention and self-attention layers
    lora_config = LoraConfig(
        r=4,                    # LoRA rank (same as default in old LoRAAttnProcessor)
        lora_alpha=4,           # scaling factor
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",   # attention projections
        ],
        lora_dropout=0.0,
    )

    # Wrap UNet with PEFT LoRA
    unet = get_peft_model(unet, lora_config)
    unet.to(device)

    # Collect only the LoRA parameters (they are the only trainable ones)
    lora_params = [p for p in unet.parameters() if p.requires_grad]

    # Print trainable parameter count for verification
    trainable = sum(p.numel() for p in lora_params)
    total = sum(p.numel() for p in unet.parameters())
    print(f"LoRA injected: {trainable:,} trainable / {total:,} total parameters ({100*trainable/total:.2f}%)")

    return unet, lora_params


def update_curve(values, label, x_label, y_label, model_path, run_id, log_steps=None):
    fig, ax = plt.subplots()
    if log_steps:
        ax.plot(log_steps, values, label=label)
    else:
        ax.plot(values, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(f'{model_path}/{label}_curve_{run_id}.png', dpi=600)
    plt.close()


def get_optimizer(parameters, config):
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=config.lr, betas=config.betas, \
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, betas=config.betas, \
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "radam":
        optimizer = torch.optim.RAdam(parameters, lr=config.lr, betas=config.betas, \
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=config.lr, momentum=config.betas[0], \
                                    weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not implemented.")
    return optimizer


def get_latents(particles, vae, rgb_as_latents=False, use_mlp_particle=False):
    if use_mlp_particle:
        images = []
        output_size = 64 if rgb_as_latents else 512
        for particle_mlp in particles:
            image = particle_mlp.generate_image(output_size)
            images.append(image)
        latents = torch.cat(images, dim=0)
        if not rgb_as_latents:
            latents = vae.config.scaling_factor * vae.encode(latents).latent_dist.sample()
    else:
        if rgb_as_latents:
            latents = F.interpolate(particles, (64, 64), mode="bilinear", align_corners=False)
        else:
            rgb_BCHW_512 = F.interpolate(particles, (512, 512), mode="bilinear", align_corners=False)
            latents = vae.config.scaling_factor * vae.encode(rgb_BCHW_512).latent_dist.sample()
    return latents


@torch.no_grad()
def batch_decode_vae(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents.clone().detach()
    bs = 8
    images = []
    for i in range(int(np.ceil(latents.shape[0] / bs))):
        batch_i = latents[i*bs:(i+1)*bs]
        image_i = vae.decode(batch_i).sample.to(torch.float32)
        images.append(image_i)
    image = torch.cat(images, dim=0)
    return image


@torch.no_grad()
def get_images(particles, vae, rgb_as_latents=False, use_mlp_particle=False):
    if use_mlp_particle:
        images = []
        output_size = 64 if rgb_as_latents else 512
        for particle_mlp in particles:
            image = particle_mlp.generate_image(output_size)
            images.append(image)
        images = torch.cat(images, dim=0)
        if rgb_as_latents:
            images = batch_decode_vae(images, vae)
    else:
        if rgb_as_latents:
            latents = F.interpolate(particles, (64, 64), mode="bilinear", align_corners=False)
            images = batch_decode_vae(latents, vae)
        else:
            images = F.interpolate(particles, (512, 512), mode="bilinear", align_corners=False)
    return images


### siren from https://github.com/vsitzmann/siren/
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, device, \
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.normal_(0, 1 / hidden_features)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)
        self.device = device
        self.out_features = out_features

    def forward(self, coords):
        output = self.net(coords)
        return output
    
    def generate_image(self, img_size=64):
        grid = torch.Tensor([[[2*(x / (img_size - 1)) - 1, 2*(y / (img_size - 1)) - 1] for y in range(img_size)] for x in range(img_size)])
        grid = grid.view(-1, 2)
        grid = grid.to(self.device)
        rgb_values = self.forward(grid)
        rgb_values = torch.tanh(rgb_values)
        rgb_values = rgb_values.view(1, img_size, img_size, self.out_features)
        image = rgb_values.permute(0, 3, 1, 2)
        return image