import logging
import os
import random

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup

# Enable TF32 for faster training
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


import wandb
from model import DiT, REPALoss
from utils import (avg_scalar_across_ranks, create_dataloader,
                   encode_prompt_with_t5, load_encoders)

CAPTURE_INPUT = False


def cleanup():
    dist.destroy_process_group()


def forward(
    dit_model,
    batch,
    vae_model,
    text_encoder,
    tokenizer,
    device,
    repa_loss,
    repa_lambda,
    global_step,
    master_process,
    ctx,
    generator=None,
    binnings=None,
):

    (images_vae, images_dinov2, captions) = batch
    images_vae = images_vae.to(device).to(torch.float32)
    images_dinov2 = images_dinov2.to(device).to(torch.float32)

    with torch.no_grad():
        vae_latent = vae_model.encode(images_vae).latent_dist.sample()
        # normalize
        vae_latent = (
            vae_latent - vae_model.config.shift_factor
        ) * vae_model.config.scaling_factor
        vae_latent = vae_latent.to(torch.bfloat16)
        caption_encoded = encode_prompt_with_t5(
            text_encoder, tokenizer, prompt=captions, device=device
        )
        caption_encoded = caption_encoded.to(torch.bfloat16)

    batch_size = images_vae.size(0)

    # log normal sample
    z = torch.randn(
        batch_size, device=device, dtype=torch.bfloat16, generator=generator
    )
    t = torch.nn.Sigmoid()(z)

    if CAPTURE_INPUT and master_process and global_step == 0:
        torch.save(vae_latent, f"test_data/vae_latent_{global_step}.pt")
        torch.save(caption_encoded, f"test_data/caption_encoded_{global_step}.pt")
        torch.save(t, f"test_data/timesteps_{global_step}.pt")

    noise = torch.randn(
        vae_latent.shape, device=device, dtype=torch.bfloat16, generator=generator
    )

    with ctx:
        # Forward pass
        z_t = vae_latent * t.reshape(batch_size, 1, 1, 1) + noise * (
            1 - t.reshape(batch_size, 1, 1, 1)
        )
        v_objective = vae_latent - noise
        output, intermediate_features = dit_model(z_t, caption_encoded, t)

        diffusion_loss_batchwise = (
            (v_objective.float() - output.float()).pow(2).mean(dim=(1, 2, 3))
        )

        diffusion_loss = diffusion_loss_batchwise.mean()
        representation_loss_batchwise = repa_loss(
            intermediate_features[0], images_dinov2
        )
        representation_loss = representation_loss_batchwise.mean()

        # timestep binning
        tbins = [int(_t * 10) for _t in t]

        if binnings is not None:
            (
                diffusion_loss_binning,
                diffusion_loss_binning_count,
                representation_loss_binning,
                representation_loss_binning_count,
            ) = binnings
            for idx, tb in enumerate(tbins):
                diffusion_loss_binning[tb] += diffusion_loss_batchwise[idx].item()
                diffusion_loss_binning_count[tb] += 1
                representation_loss_binning[tb] += representation_loss_batchwise[
                    idx
                ].item()
                representation_loss_binning_count[tb] += 1

        # Combine losses
        total_loss = diffusion_loss + repa_lambda * representation_loss

    return total_loss, diffusion_loss, representation_loss


@click.command()
@click.option(
    "--dataset_url",
    type=str,
    default="/home/ubuntu/pd12m.int8/dataset/cc12m-wds/cc12m-train-{0000..2160}.tar",
    help="URL for training dataset",
)
@click.option(
    "--test_dataset_url",
    type=str,
    default="/home/ubuntu/pd12m.int8/dataset/cc12m-wds/cc12m-train-{2161..2168}.tar",
    help="URL for test dataset",
)
@click.option("--num_epochs", type=int, default=2, help="Number of training epochs")
@click.option("--batch_size", type=int, default=64, help="Batch size for training")
@click.option("--learning_rate", type=float, default=1e-4, help="Learning rate")
@click.option("--max_steps", type=int, default=10000, help="Maximum training steps")
@click.option(
    "--evaluate_every", type=int, default=20, help="Steps between evaluations"
)
@click.option(
    "--alignment_layer", type=int, default=8, help="Which layer to align in REPA"
)
@click.option("--repa_lambda", type=float, default=0.5, help="Weight for REPA loss")
@click.option("--run_name", type=str, default="diffusion_repa", help="Name of run")
@click.option("--model_width", type=int, default=512, help="Width of the model")
@click.option("--model_depth", type=int, default=9, help="Depth of the model")
@click.option(
    "--model_head_dim", type=int, default=128, help="Head dimension of the model"
)
@click.option("--compile_models", type=bool, default=False, help="Compile models")
def train_ddp(
    dataset_url,
    test_dataset_url,
    num_epochs,
    batch_size,
    learning_rate,
    max_steps,
    evaluate_every,
    alignment_layer,
    repa_lambda,
    run_name,
    model_width,
    model_depth,
    model_head_dim,
    compile_models,
):
    # Initialize distributed training
    assert torch.cuda.is_available(), "CUDA is required for training"
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Initialize wandb for the master process

    vae_model, tokenizer, text_encoder = load_encoders(device=device, compile_models=compile_models)

    dit_model = DiT(
        in_channels=16,
        patch_size=2,
        depth=model_depth,
        num_heads=model_width // model_head_dim,
        mlp_ratio=4.0,
        cross_attn_input_size=4096,
        hidden_size=model_width,
        repa_target_layers=[6],
        repa_target_dim=1536,
    ).to(device)

    if master_process:
        param_count = sum(p.numel() for p in dit_model.parameters())

        wandb.init(
            project="diffusion_repa",
            name=run_name,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "alignment_layer": alignment_layer,
                "repa_lambda": repa_lambda,
                "model_parameters": param_count / 1e6,
                "model_width": model_width,
                "model_depth": model_depth,
                "model_head_dim": model_head_dim,
            },
        )

    # Wrap model in DDP
    dit_model = DDP(dit_model, device_ids=[ddp_rank])
    
    if compile_models:
        dit_model.forward = torch.compile(dit_model.forward, mode="reduce-overhead")

    # Initialize optimizer and scheduler
    param_setup = []
    constant_param_name = ["patch_proj", "context_kv", "positional_embedding"]

    optimizer_grouped_parameters, final_optimizer_settings = (
        dit_model.module.get_mup_setup(learning_rate, 1e-3, constant_param_name)
    )

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.95),
        fused=True,
    )

    num_warmup_steps = max_steps // 10
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, max_steps
    )

    # Create dataloaders
    train_loader = create_dataloader(dataset_url, batch_size, num_workers=8)
    test_loader = create_dataloader(
        test_dataset_url, batch_size, num_workers=1, just_resize=True, do_shuffle=False
    )

    # Initialize loss functions
    repa_loss = REPALoss(dtype=torch.float32).to(device)

    # Setup automatic mixed precision
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if master_process:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Initialize step counter
    global_step = 0

    # Training loop
    dit_model.train()

    diffusion_loss_binning = {k: 0 for k in range(10)}
    diffusion_loss_binning_count = {k: 0 for k in range(10)}
    representation_loss_binning = {k: 0 for k in range(10)}
    representation_loss_binning_count = {k: 0 for k in range(10)}

    for epoch in range(num_epochs):
        if global_step >= max_steps:
            break

        for batch_idx, batch in enumerate(train_loader):

            if global_step >= max_steps:
                break

            total_loss, diffusion_loss, representation_loss = forward(
                dit_model,
                batch,
                vae_model,
                text_encoder,
                tokenizer,
                device,
                repa_loss,
                repa_lambda,
                global_step,
                master_process,
                ctx,
                binnings=(
                    diffusion_loss_binning,
                    diffusion_loss_binning_count,
                    representation_loss_binning,
                    representation_loss_binning_count,
                ),
            )
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Logging
            if global_step % 10 == 0:

                diffusion_loss = avg_scalar_across_ranks(diffusion_loss.item())
                representation_loss = avg_scalar_across_ranks(
                    representation_loss.item()
                )
                total_loss = avg_scalar_across_ranks(total_loss.item())

                if master_process:
                    wandb.log(
                        {
                            "train/diffusion_loss": diffusion_loss,
                            "train/representation_loss": representation_loss,
                            "train/total_loss": total_loss,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": global_step,
                            "train_binning/diffusion_loss_binning": {
                                k: v / max(c, 1)
                                for k, v, c in zip(
                                    diffusion_loss_binning.keys(),
                                    diffusion_loss_binning.values(),
                                    diffusion_loss_binning_count.values(),
                                )
                            },
                            "train_binning/representation_loss_binning": {
                                k: v / max(c, 1)
                                for k, v, c in zip(
                                    representation_loss_binning.keys(),
                                    representation_loss_binning.values(),
                                    representation_loss_binning_count.values(),
                                )
                            },
                        }
                    )

                    diffusion_per_timestep = "\n\t".join(
                        [
                            f"{k}: {v / max(c, 1):.4f}"
                            for k, v, c in zip(
                                diffusion_loss_binning.keys(),
                                diffusion_loss_binning.values(),
                                diffusion_loss_binning_count.values(),
                            )
                        ]
                    )
                    repa_per_timestep = "\n\t".join(
                        [
                            f"{k}: {v / max(c, 1):.4f}"
                            for k, v, c in zip(
                                representation_loss_binning.keys(),
                                representation_loss_binning.values(),
                                representation_loss_binning_count.values(),
                            )
                        ]
                    )

                    logger.info(
                        f"Epoch [{epoch}/{num_epochs}] "
                        f"Step [{global_step}/{max_steps}] "
                        f"Loss: {total_loss:.4f} "
                        f"(Diff: {diffusion_loss:.4f}, "
                        f"REPA: {representation_loss:.4f})"
                        f"LR: {lr_scheduler.get_last_lr()[0]:.4f}"
                        f"\nDiffusion Per-timestep-binned:\n{diffusion_per_timestep}"
                        f"\nREPA Per-timestep-binned:\n{repa_per_timestep}"
                    )
                    diffusion_loss_binning = {k: 0 for k in range(10)}
                    diffusion_loss_binning_count = {k: 0 for k in range(10)}
                    representation_loss_binning = {k: 0 for k in range(10)}
                    representation_loss_binning_count = {k: 0 for k in range(10)}

            global_step += 1

            if global_step % evaluate_every == 1:
                # evaluate(dit_model, test_loader, device)
                generator = torch.Generator(device=device).manual_seed(42)

                val_diffusion_loss_binning = {k: 0 for k in range(10)}
                val_diffusion_loss_binning_count = {k: 0 for k in range(10)}
                val_representation_loss_binning = {k: 0 for k in range(10)}
                val_representation_loss_binning_count = {k: 0 for k in range(10)}

                dit_model.eval()

                total_losses = []
                diffusion_losses = []
                representation_losses = []

                for batch_idx, batch in enumerate(test_loader):
                    with torch.no_grad():

                        total_loss, diffusion_loss, representation_loss = forward(
                            dit_model,
                            batch,
                            vae_model,
                            text_encoder,
                            tokenizer,
                            device,
                            repa_loss,
                            repa_lambda,
                            global_step,
                            master_process,
                            ctx,
                            generator,
                            binnings=(
                                val_diffusion_loss_binning,
                                val_diffusion_loss_binning_count,
                                val_representation_loss_binning,
                                val_representation_loss_binning_count,
                            ),
                        )

                        total_losses.append(total_loss.item())
                        diffusion_losses.append(diffusion_loss.item())
                        representation_losses.append(representation_loss.item())

                    if batch_idx == 40:
                        break

                dit_model.train()
                logger.info(
                    f"Saving checkpoint to checkpoints/{run_name}/{global_step}.pt"
                )

                total_loss = avg_scalar_across_ranks(np.mean(total_losses).item())
                diffusion_loss = avg_scalar_across_ranks(
                    np.mean(diffusion_losses).item()
                )
                representation_loss = avg_scalar_across_ranks(
                    np.mean(representation_losses).item()
                )

                if master_process:
                    os.makedirs("checkpoints", exist_ok=True)
                    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
                    torch.save(
                        dit_model.module.state_dict(),
                        f"checkpoints/{run_name}/{global_step}.pt",
                    )
                    wandb.log(
                        {
                            "test/total_loss": total_loss,
                            "test/diffusion_loss": diffusion_loss,
                            "test/representation_loss": representation_loss,
                            "test_binning/diffusion_loss_binning": {
                                k: v / max(c, 1)
                                for k, v, c in zip(
                                    val_diffusion_loss_binning.keys(),
                                    val_diffusion_loss_binning.values(),
                                    val_diffusion_loss_binning_count.values(),
                                )
                            },
                            "test_binning/representation_loss_binning": {
                                k: v / max(c, 1)
                                for k, v, c in zip(
                                    val_representation_loss_binning.keys(),
                                    val_representation_loss_binning.values(),
                                    val_representation_loss_binning_count.values(),
                                )
                            },
                        }
                    )

    # Cleanup
    if master_process:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    train_ddp()
