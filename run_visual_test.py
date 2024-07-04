import argparse
import json
import pytorch_lightning
import pytorch_lightning.callbacks
import pytorch_lightning.loggers
import torch
import torchvision
import torchvision.ops
import torchvision.transforms
import torchvision.transforms.v2
import torchvision.transforms.v2.functional
import torchvision.utils

import avit


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--ckpt", action="store", required=True
    )
    cli.add_argument(
        "--img_idxs", action="store", nargs="+", type=int,
        default=[22, 29, 40, 43, 76, 109]
    )
    args = cli.parse_args()

    pytorch_lightning.seed_everything(seed=avit.SEED, workers=True)

    print("Instantiating the Caltech-256 data module...")
    dm = avit.Caltech256DataModule(avit.HF_MODEL_NAME, 128)
    dm.prepare_data()
    dm.setup("test")
    print("Caltech-256 data module instantiated.")

    print("Instantiating the model...")
    model = avit.AViTModel.load_from_checkpoint(
        args.ckpt, avit_kwargs=avit.AVIT_KWARGS, map_location="cpu"
    )
    model = model.eval().requires_grad_(requires_grad=False)
    print("Model instantiated.")

    print("Instantiating auxiliary variables...")
    collator = avit.ImageClassificationDataCollator()
    imagenet_mean = torch.tensor(avit.IMAGENET_MEAN)
    imagenet_std = torch.tensor(avit.IMAGENET_STD)
    with open("assets/idx2label.json", mode="r", encoding="utf-8") as f:
        obj = f.read()
    idx2label = json.loads(obj)
    print("Auxiliary variables instantiated.")

    print("Building the test batch of images...")
    top_imgs_idxs = args.img_idxs
    top_data_points = list(map(lambda idx: dm.test_ds[idx], top_imgs_idxs))
    top_batch = collator(top_data_points)
    print("Testi batch of images built.")

    print("Performing inference...")
    outputs = model(top_batch["pixel_values"])
    preds = torch.argmax(outputs["logits"], dim=-1)
    batch_counter = outputs["counter"].cpu().detach()
    print("Inference performed.")

    print("Saving the results (assets/results/visual_test.png)...")
    imgs_to_show = list()
    for i in range(top_batch["labels"].shape[0]):
        counter = batch_counter[i, 1:]
        counter = counter.reshape((14, 14))
        img = top_batch["pixel_values"][i]
        img = torchvision.transforms.v2.functional.normalize(
            img,
            -imagenet_mean / imagenet_std,
            torch.ones_like(imagenet_std) / imagenet_std
        )
        counter = counter.expand(3, 14, 14)
        counter = torchvision.transforms.v2.functional.resize(
            counter, [224, 224],
            interpolation=torchvision.transforms.v2.InterpolationMode.NEAREST
        )
        mask = counter == 12.
        masked_img1 = img * counter
        masked_img1 = masked_img1 / masked_img1.max()
        masked_img2 = img * mask.float()
        counter = counter / counter.max()
        img = torchvision.utils.draw_bounding_boxes(
            (img * 255.).to(dtype=torch.uint8),
            torchvision.ops.masks_to_boxes(torch.ones((1, 224, 224), dtype=torch.bool)),
            labels=[
                "true: " + \
                str(idx2label[str(top_batch['labels'][i].item())]) + \
                "\npred: " + str(idx2label[str(preds[i].item())])
            ],
            colors=["#FE0AE2"], width=0, font="assets/KFOlCnqEu92Fr1MmEU9vAw.ttf",
            font_size=20
        )
        img = img / img.max()

        imgs_to_show.extend([img, counter, masked_img1, mask, masked_img2])
    
    torchvision.transforms.v2.functional.to_pil_image(
        torchvision.utils.make_grid(imgs_to_show, nrow=5)
    ).save("assets/results/visual_test.png")
    print("Results saved.")
