# aerial2map

In this repo, you’ll find a project to build a generative model based on the paper [*Image-to-Image Translation with Conditional Adversarial Networks*](https://arxiv.org/abs/1611.07004) by Isola et al., 2017, commonly known as Pix2Pix.

Here, we are training a model to convert aerial satellite images (input) into map routes (output), just as shown in the original paper. The generator architecture is based on U-Net, while the discriminator uses PatchGAN. The primary focus is on setting up the loss functions that enable the model to balance between reconstructing accurate shapes and creating realistic map details. You can either start training from a pre-trained model checkpoint or begin training from scratch.



---

## Learning Objectives

1. **Implement Pix2Pix Loss:** Understand and implement the unique loss function that sets Pix2Pix apart from a standard supervised U-Net.
2. **Observe Model Progression:** Track how the Pix2Pix generator’s priorities evolve as it trains, shifting from reconstruction accuracy to improved realism.

---

## Project Overview

This project aims to generate maps from satellite images using the Pix2Pix GAN architecture. The generator follows a U-Net structure, while the discriminator is a PatchGAN. The model checkpoint (pix2pix_15000.pth) saves the current training state, which you can use as a starting point. However, training can also be done entirely from scratch if desired.

---

## Training Parameters

Below are the key parameters used for training:

- **Loss Functions:**
  - `adv_criterion = nn.BCEWithLogitsLoss()`
  - `recon_criterion = nn.L1Loss()`
  - `lambda_recon = 200` (weight for reconstruction loss)

- **Training Setup:**
  - `n_epochs = 20`
  - `input_dim = 3` (for RGB images)
  - `real_dim = 3` (for RGB maps)
  - `display_step = 200` (to visualize progress every 200 steps)
  - `batch_size = 4`
  - `lr = 0.0002` (learning rate)
  - `target_shape = 256 x 256` (input and output image size)
  - `device = 'cpu'` (or `cuda` if available on your device)

---

## Model Dimensions

- **Input Tensor Shape:** `torch.Size([1, 3, 256, 256])`
- **Output Tensor Shape:** `torch.Size([1, 3, 256, 256])`

---

## Contributing

This project is open to contributions! Feel free to submit pull requests or issues.

---

## License

This project is licensed under the MIT License.