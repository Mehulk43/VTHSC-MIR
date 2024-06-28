# VTHSC-MIR PRL 2024
VTHSC-MIR: Vision Transformer Hashing with Supervised Contrastive learning based medical image retrieval 
{https://doi.org/10.1016/j.patrec.2024.06.003)



## How to Run

This code uses the Vision Transformer (ViT) code and pretrained model (https://github.com/jeonsworld/ViT-pytorch) (https://github.com/shivram1987/VisionTransformerHashing) and DeepHash framework (https://github.com/swuxyj/DeepHash-pytorch).

Download the ViT pretrained models from official repository and keep under pretrainedVIT directory:

ViT-B_16: https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz 

ViT-B_32: https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz

Download data from https://github.com/swuxyj/DeepHash-pytorch for different dataset, if not already present under data directory.


### Paper Citation
Please cite following paper if you make use of this code in your research:

@article{kumar2024vthsc,
  title={VTHSC-MIR: Vision Transformer Hashing with Supervised Contrastive learning based medical image retrieval},
  author={Kumar, Mehul and Singh, Rhythumwinder and Mukherjee, Prerana},
  journal={Pattern Recognition Letters},
  year={2024},
  publisher={Elsevier}
}
