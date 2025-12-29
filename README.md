# The Prism Hypothesis: Harmonizing Semantic and Pixel Representations via Unified Autoencoding

<div class="is-size-5 publication-authors", align="center">
              <!-- Paper authors -->
                <span class="author-block">
                  <a href="https://weichenfan.github.io/Weichen//" target="_blank">Weichen Fan</a><sup>1</sup>,</span>
                  <span class="author-block">
                    <a href="https://paranioar.github.io/" target="_blank">Haiwen Diao</a><sup>1</sup>,</span>
                  <span class="author-block">
                  <a href="https://openreview.net/profile?id=~Quan_Wang6" target="_blank">Quan Wang</a><sup>2</sup>,</span>
                  <span class="author-block">
                  <a href="http://dahua.site/" target="_blank">Dahua Lin</a><sup>2</sup>,</span>
                  <span class="author-block">
                    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>1✉</sup>
                  </span>
                  </div>
<div class="is-size-5 publication-authors", align="center">
                    <span class="author-block">S-Lab, Nanyang Technological University<sup>1</sup> &nbsp;&nbsp;&nbsp;&nbsp; SenseTime Research <sup>2</sup> </span>
                    <span class="eql-cntrb"><small><br><sup>✉</sup>Corresponding Author.</small></span>
                  </div>

</p>

<div align="center">
                      <a href="https://arxiv.org/abs/2512.19693">Paper</a> | <a href="https://huggingface.co/weepiess2383/UAE">Weights</a> 
                      <!-- <a href="https://weichenfan.github.io/webpage-cfg-zero-star/">Project Page</a> |
                      <a href="https://huggingface.co/spaces/weepiess2383/CFG-Zero-Star">Demo</a> |
                      <a href="https://huggingface.co/spaces/jamesliu1217/EasyControl_Ghibli">Demo for Ghibli style</a> -->
</div>

---

<!-- ![](https://img.shields.io/badge/Vchitect2.0-v0.1-darkcyan)
![](https://img.shields.io/github/stars/Vchitect/Vchitect-2.0)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVchitect-2.0&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Generic badge](https://img.shields.io/badge/DEMO-Vchitect2.0_Demo-<COLOR>.svg)](https://huggingface.co/spaces/Vchitect/Vchitect-2.0)
[![Generic badge](https://img.shields.io/badge/Checkpoint-red.svg)](https://huggingface.co/Vchitect/Vchitect-XL-2B) -->
🔥 More coming soon!

## Installation
~~~bash
conda create -n uae python=3.10 -y
conda activate uae
pip install uv

uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121

uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb
uv pip install "numpy<2" transformers einops omegaconf
uv pip install torchmetrics
~~~
## Quick evaluation
~~~ bash
python eval_uae.py \
  --config unified_ae/configs/stage1_infer.yaml \
  --checkpoint PATH_TO_WEIGHTS \
  --imagenet-path PATH_TO_IMAGENET \
  --coco-path PATH_TO_COCO \
  --batch-size 16 \
  --num-workers 8 \
  --image-size 256 \
  --freq-ratio 1.0 \
  --log-file logs/uae_eval_metrics.txt
~~~

**Expected Results:**
~~~bash
ImageNet: PSNR=29.588 dB | SSIM=0.8789 | rFID=0.193
MS-COCO: PSNR=29.484 dB | SSIM=0.8846 | rFID=0.157
~~~

## Training
There are four sub-stages to train our UAE model.

Follow the scripts to step-by-step reproduce our results.
~~~ bash
# sub-stage 1
export WANDB_API_KEY=YOUR_KEY
export WANDB_ENTITY=YOUR_ID
export WANDB_PROJECT=PROJECT_NAME

DATA_ROOT=PATH_TO_TRAIN_OF_IMGNET
VAL_ROOT=PATH_TO_VAL_OF_IMGNET                              

accelerate launch train_uae.py \
  --config unified_ae/configs/stage1_train.yaml \
  --stage-key sub_stage1 \
  --data-path "$DATA_ROOT" \
  --val-path "$VAL_ROOT" \
  --results-dir results/sub_stage1 \
  --mixed-precision YOUR_PRECISION(bf16 or no) \
  --wandb --wandb-name uae_1
~~~
After this you will get model with FID=103.870, PSNR=18.025 dB

~~~ bash
# sub-stage 2
export WANDB_API_KEY=YOUR_KEY
export WANDB_ENTITY=YOUR_ID
export WANDB_PROJECT=PROJECT_NAME

DATA_ROOT=PATH_TO_TRAIN_OF_IMGNET
VAL_ROOT=PATH_TO_VAL_OF_IMGNET  

accelerate launch train_uae.py \
  --config unified_ae/configs/stage1_train.yaml \
  --stage-key sub_stage2 \
  --data-path "$DATA_ROOT" \
  --val-path "$VAL_ROOT" \
  --results-dir results/sub_stage2 \
  --mixed-precision YOUR_PRECISION(bf16 or no) \
  --wandb --wandb-name uae_2
~~~
After this you will get model with FID=0.968, PSNR=27.356 dB

~~~ bash
# sub-stage 3
export WANDB_API_KEY=YOUR_KEY
export WANDB_ENTITY=YOUR_ID
export WANDB_PROJECT=PROJECT_NAME

DATA_ROOT=PATH_TO_TRAIN_OF_IMGNET
VAL_ROOT=PATH_TO_VAL_OF_IMGNET  

accelerate launch train_uae.py \
  --config unified_ae/configs/stage1_train.yaml \
  --stage-key sub_stage3 \
  --data-path "$DATA_ROOT" \
  --val-path "$VAL_ROOT" \
  --results-dir results/sub_stage3 \
  --mixed-precision YOUR_PRECISION(bf16 or no) \
  --wandb --wandb-name uae_3
~~~
After this you will get model with FID=0.530, PSNR=30.110 dB

~~~ bash
# sub-stage 4
export WANDB_API_KEY=YOUR_KEY
export WANDB_ENTITY=YOUR_ID
export WANDB_PROJECT=PROJECT_NAME

DATA_ROOT=PATH_TO_TRAIN_OF_IMGNET
VAL_ROOT=PATH_TO_VAL_OF_IMGNET  

accelerate launch train_uae.py \
  --config unified_ae/configs/stage1_train.yaml \
  --stage-key sub_stage4 \
  --data-path "$DATA_ROOT" \
  --val-path "$VAL_ROOT" \
  --results-dir results/sub_stage4 \
  --mixed-precision YOUR_PRECISION(bf16 or no) \
  --wandb --wandb-name uae_4
~~~
After this you will get model with FID=0.166, PSNR=29.499 dB


## BibTex
```
@misc{fan2025uae,
      title={The Prism Hypothesis: Harmonizing Semantic and Pixel Representations via Unified Autoencoding}, 
      author={Weichen Fan and Haiwen Diao and Quan Wang and Dahua Lin and Ziwei Liu},
      year={2025},
      eprint={2512.19693},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.19693}, 
}
```

## ✨ Star History


[![Star History Chart](https://api.star-history.com/svg?repos=WeichenFan/UAE&type=date&legend=top-left)](https://www.star-history.com/#WeichenFan/UAE&type=date&legend=top-left)