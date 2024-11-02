import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import webdataset as wds
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, T5TokenizerFast


def avg_scalar_across_ranks(scalar):
    world_size = dist.get_world_size()
    scalar_tensor = torch.tensor(scalar, device="cuda")
    dist.all_reduce(scalar_tensor, op=dist.ReduceOp.AVG)
    return scalar_tensor.item()


def create_dataloader(url, batch_size, num_workers, do_shuffle=True, just_resize=False):
    dataset = wds.WebDataset(
        url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker
    )
    dataset = dataset.shuffle(1000) if do_shuffle else dataset

    transform_for_vae = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((256, 256)),
        ]
    )

    transform_for_dinov2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
        ]
    )

    def transform_data(data):
        image, json_data = data
        image_vae = transform_for_vae(image)
        image_dinov2 = transform_for_dinov2(image)
        return image_vae, image_dinov2, json_data["caption"]

    dataset = (
        dataset.decode("rgb")
        .to_tuple("jpg;png", "json", handler=wds.warn_and_continue)
        .map(transform_data)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return loader


def encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def load_encoders(
    vae_path="black-forest-labs/FLUX.1-dev",
    text_encoder_path="black-forest-labs/FLUX.1-dev",
    device="cuda",
    compile_models=True,
):

    vae_model = (
        AutoencoderKL.from_pretrained(
            vae_path, torch_dtype=torch.float32, subfolder="vae"
        )
        .to(device)
        .eval()
    )
    vae_model.to(memory_format=torch.channels_last)
    vae_model.requires_grad_(False)

    tokenizer = T5TokenizerFast.from_pretrained(
        text_encoder_path, subfolder="tokenizer_2"
    )

    text_encoder = (
        T5EncoderModel.from_pretrained(
            text_encoder_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        .to(device)
        .eval()
    )
    text_encoder.requires_grad_(False)

    if compile_models:
        vae_model.encode = torch.compile(vae_model.encode, mode="reduce-overhead")
        text_encoder.forward = torch.compile(
            text_encoder.forward, mode="reduce-overhead"
        )

    return vae_model, tokenizer, text_encoder


import os

# test pipe
import torchvision


def test_pipe(
    dataset_path="/home/ubuntu/pd12m.int8/dataset/cc12m-wds/cc12m-train-0009.tar",
    device="cuda:1",
):
    vae_model, tokenizer, text_encoder = load_encoders(device=device)
    dataloader = create_dataloader(
        dataset_path, batch_size=4, num_workers=1, do_shuffle=False, just_resize=True
    )

    for batch in dataloader:
        images_vae, images_dinov2, captions = batch
        images_vae = images_vae.to(device, dtype=torch.float16)
        images_dinov2 = images_dinov2.to(device, dtype=torch.float16)
        print(images_vae.shape, images_dinov2.shape, captions)

        prompt_embeds = encode_prompt_with_t5(
            text_encoder, tokenizer, prompt=captions, device=device
        )
        vae_latent = vae_model.encode(images_vae).latent_dist.sample()

        print(
            vae_latent.shape,
            prompt_embeds.shape,
            vae_latent.device,
            prompt_embeds.device,
        )

        # write it to ./test_data folder
        os.makedirs("./test_data", exist_ok=True)
        torch.save(vae_latent, f"./test_data/vae_latent.pt")
        torch.save(prompt_embeds, f"./test_data/prompt_embeds.pt")

        # unnormalize images and write to ./test_data folder
        images_vae = images_vae * 0.5 + 0.5

        images_dinov2 = images_dinov2.cpu()
        images_dinov2 = images_dinov2 * torch.tensor([0.229, 0.224, 0.225]).reshape(
            1, 3, 1, 1
        ) + torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)

        torchvision.utils.save_image(images_vae, f"./test_data/images_vae.png")
        torchvision.utils.save_image(images_dinov2, f"./test_data/images_dinov2.png")

        break


if __name__ == "__main__":
    test_pipe()
