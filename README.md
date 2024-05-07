# StereoDiffusionM: COS429 Final Project

## Existing image to Stereo image generation
(section adapted from StereoDiffusion's README)
cd into the StereoDiffusion folder. We use DPT depth model, so you need to download this model first. You can download the DPT depth model `dpt_hybrid-midas-501f0c75.pt` from [here](https://github.com/isl-org/DPT). Then, you need to put DPT depth model in `midas_models` folder.

- clone [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [stablediffusion](https://github.com/Stability-AI/stablediffusion) and [DPT](https://github.com/isl-org/DPT)
- run `pip install -r prompt-to-prompt/requirements.txt` to install the required packages.
- run `python img2stereo.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt  --img_path="path/to/your/image"` , or follow jobD.slurm

## Setting up Marigold
cd into StereoDiffusion folder
- clone [Marigold](https://github.com/prs-eth/marigold) and follow the setup instructions on the Marigold github
- Move the file "runMT.py" from Marigold_tmp to the newly cloned Marigold folder. Delete Marigold_tmp
- to run: follow jobM.slurm

## Setting up testing
cd into TestStereoDiffusion
- clone [SSIM-Pytorch](https://github.com/richzhang/PerceptualSimilarity.git)