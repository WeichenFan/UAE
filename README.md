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
🔥 On the way

## Quick evaluation
~~~ bash
python eval_unified_ae.py \
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
<!-- 🔥 [Huggingface demo for Ghibli style generation](https://huggingface.co/spaces/jamesliu1217/EasyControl_Ghibli) supported by [EasyControl](https://github.com/Xiaojiu-z/EasyControl).

⚡️ [Huggingface demo](https://huggingface.co/spaces/weepiess2383/CFG-Zero-Star) now supports text-to-image generation with SD3 and SD3.5. -->
