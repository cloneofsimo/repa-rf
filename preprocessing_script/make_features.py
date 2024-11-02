import argparse
import json
import logging
import os
import re
import time
from typing import Any

import numpy as np
import torch
import webdataset as wds
from diffusers.models import AutoencoderKL
from PIL import Image
from streaming import MDSWriter
from streaming.base.format.mds.encodings import Encoding, _encodings
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer,
                          BitsAndBytesConfig, CLIPTextModel, CLIPTokenizer,
                          SiglipModel, T5EncoderModel, T5TokenizerFast)

# Initialize logging
logging.basicConfig(level=logging.INFO)


def modify_caption(caption: str) -> str:
    cap = caption.replace("This image displays", "").strip()
    if cap.startswith(":"):
        cap = cap[1:]
    return cap.strip()


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.uint8)


class np16(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float16)


_encodings["np16"] = np16
_encodings["uint8"] = uint8

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose(
    [
        transforms.Resize(288),
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        normalize,
    ]
)


def crop_to_center(image, new_size=768):
    width, height = image.size
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def prepare_image(pil_image):
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image


def wds_preprocess(x):
    key, pil_image, _json = x
    pil_image = pil_image.convert("RGB")
    if pil_image.size[0] > pil_image.size[1]:
        pil_image = pil_image.resize(
            (int(pil_image.size[0] * 512 / pil_image.size[1]), 512)
        )
    else:
        pil_image = pil_image.resize(
            (512, int(pil_image.size[1] * 512 / pil_image.size[0]))
        )
    pil_image = crop_to_center(pil_image, new_size=512)

    image_for_vae = prepare_image(pil_image)

    caption = _json["caption"] or ""
    uid = _json.get("uid", key)
    watermark_class = _json.get("watermark_class_id", 1)
    est = _json.get("aesthetic_score", 100)
    # image_for_vae, captions, uids, est, pil_images = batch

    return (image_for_vae, caption, uid, est, [pil_image])


COLUMNS = {
    "key": "str",
    "caption": "str",
    "vae_512x512_latents": "np16",
    "t5_emb": "np16",
    "clip_emb": "np16",
    "hps_score": "str",
    "siglip_text_vec": "np16",
    "siglip_image_vec": "np16",
    "siglip_sim": "float32",
}


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(
        device=device, dtype=dtype
    )
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


@torch.no_grad()
def convert_to_mds(dataset_paths, out_roots, device, is_test=False):
    logging.info(f"Processing on {device}")

    pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
    vae_model = (
        AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.float16, subfolder="vae"
        )
        .to(device)
        .eval()
    )
    vae_model.to(memory_format=torch.channels_last)

    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer_2"
    )

    text_encoder_one = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one.to(device), text_encoder_two.to(device)]

    # sscd_model = torch.jit.load("sscd_disc_mixup.torchscript.pt").to(device)

    siglip_model = (
        SiglipModel.from_pretrained("google/siglip-large-patch16-256").to(device).eval()
    )
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-256")

    num_chunk = 8

    dataset_bulks = [
        dataset_paths[i : i + num_chunk]
        for i in range(0, len(dataset_paths), num_chunk)
    ]
    out_roots_bulks = [
        out_roots[i : i + num_chunk] for i in range(0, len(out_roots), num_chunk)
    ]

    for dataset_paths, out_roots in zip(dataset_bulks, out_roots_bulks):
        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                logging.info(f"Dataset not found: {dataset_path}")
                return

        out_root = out_roots[0]

        dataset = wds.DataPipeline(
            wds.SimpleShardList(dataset_paths),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.decode("pil", handler=wds.warn_and_continue),
            wds.to_tuple("__key__", "jpg;png", "json", handler=wds.warn_and_continue),
            wds.map(wds_preprocess),
            wds.batched(64),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=16,
            prefetch_factor=4,
            shuffle=False,
            drop_last=False,
        )

        t0 = time.time()
        sub_data_root = os.path.join(out_root, "data")

        if os.path.exists(sub_data_root):
            for file in os.listdir(sub_data_root):
                os.remove(os.path.join(sub_data_root, file))

        os.makedirs(sub_data_root, exist_ok=True)
        inference_latencies = []
        keys = []

        with MDSWriter(out=sub_data_root, columns=COLUMNS) as out:
            for idx, batch in tqdm(enumerate(dataloader)):
                if is_test and idx > 0:
                    break

                start_time = time.time()

                image_for_vae, captions, uids, est, pil_images = batch
                pil_images = [pil_images[i][0] for i in range(len(pil_images))]

                est_idx = np.where(np.array(est) > 3)[0]

                if len(est_idx) == 0:
                    continue

                image_for_vae = image_for_vae[est_idx]
                uids = [uids[i] for i in est_idx]
                est = np.array(est)[est_idx]

                # VAE
                image_for_vae = image_for_vae.to(device).half()
                vae_latents = vae_model.encode(image_for_vae).latent_dist.sample()
                vae_outputs = vae_latents.cpu().numpy().astype(np.float16)

                prompt_embeds, pooled_prompt_embeds, _ = encode_prompt(
                    text_encoders,
                    tokenizers,
                    captions,
                    max_sequence_length=512,
                    device=device,
                    num_images_per_prompt=1,
                    text_input_ids_list=None,
                )

                # print(pooled_prompt_embeds.shape, prompt_embeds.shape, vae_outputs.shape, image_for_vae.shape)

                assert pooled_prompt_embeds.shape[0] == len(captions)
                assert prompt_embeds.shape[0] == len(captions)
                assert vae_outputs.shape[0] == len(captions)
                assert image_for_vae.shape[0] == len(captions)

                # SigLIP
                siglip_inputs = siglip_processor(
                    text=captions,
                    images=pil_images,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                ).to(device)

                siglip_outputs = siglip_model(**siglip_inputs)

                siglip_text_embeddings = (
                    siglip_outputs.text_embeds.cpu().numpy().astype(np.float16)
                )
                siglip_image_embeddings = (
                    siglip_outputs.image_embeds.cpu().numpy().astype(np.float16)
                )

                # elementwise cos similarity
                # normalize first

                siglip_similarities = np.einsum(
                    "ij,ij->i", siglip_text_embeddings, siglip_image_embeddings
                )

                # Write
                for i in range(len(captions)):
                    if siglip_similarities[i] < 0.05:
                        print("Oh no, not similar!")
                        # # write the image and caption as json, img at ./local_bad_images
                        # os.makedirs("./local_bad_images", exist_ok=True)
                        # pil_images[i].save(f"./local_bad_images/{uids[i]}.jpg")
                        # with open(f"./local_bad_images/{uids[i]}.json", "w") as f:
                        #     json.dump({"caption": captions[i], "similarity": float(siglip_similarities[i])}, f)
                        continue

                    sample = {
                        "vae_512x512_latents": vae_outputs[i],
                        "caption": str(captions[i]),
                        "clip_emb": pooled_prompt_embeds[i]
                        .cpu()
                        .numpy()
                        .astype(np.float16),
                        "t5_emb": prompt_embeds[i].cpu().numpy().astype(np.float16),
                        "key": uids[i],
                        "hps_score": str(est[i]),
                        "siglip_text_vec": siglip_text_embeddings[i],
                        "siglip_image_vec": siglip_image_embeddings[i],
                        "siglip_sim": float(siglip_similarities[i]),
                    }
                    out.write(sample)

                inference_latencies.append(time.time() - start_time)
                keys.extend(uids)

            logging.info(
                f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds"
            )
            logging.info(
                f"Total Inference Time on {device}: {time.time() - t0} seconds"
            )

        save_to_json(keys, os.path.join(out_root, "keys.json"))


def main(datasetinfos, out_roots, is_test=False, device_name="cuda"):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Processing on {device}")
    convert_to_mds(datasetinfos, out_roots, device, is_test=is_test)
    logging.info("Finished processing images.")


def detect_small_or_nonexistent_dirs(current_dir, start=0, end=18503, max_size=512):
    small_or_nonexistent_dirs = []

    for i in range(start, end + 1):
        dir_name = f"{i:05d}"
        dir_path = os.path.join(current_dir, dir_name)

        if not os.path.exists(dir_path):
            if i % 64 < 8:
                small_or_nonexistent_dirs.append(i)
        elif os.path.isdir(dir_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)

            if total_size < max_size:
                small_or_nonexistent_dirs.append(i)

    return small_or_nonexistent_dirs


def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to MDS format.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for processing (cuda or cpu).",
    )
    parser.add_argument(
        "--file_index", type=int, default=0, help="File index to process."
    )
    parser.add_argument(
        "--is_test", action="store_true", help="Run in test mode with reduced dataset."
    )
    parser.add_argument(
        "--outdir_basepath",
        type=str,
        default="../dataset/mds_pp",
        help="Output directory path.",
    )
    parser.add_argument(
        "--tar_indir_basepath",
        type=str,
        default="../dataset/art_webdataset",
        help="Input directory path.",
    )

    args = parser.parse_args()

    reqsids = list(range(2000))

    out_roots, datasetinfos = [], []
    for i, reqid in enumerate(reqsids):
        if i % 8 == args.file_index:
            out_root = f"{args.outdir_basepath}/{str(int(reqid)).zfill(5)}"
            dataset_path = f"{args.tar_indir_basepath}/{str(int(reqid)).zfill(5)}.tar"
            out_roots.append(out_root)
            datasetinfos.append(dataset_path)

    main(datasetinfos, out_roots, is_test=args.is_test, device_name=args.device)
