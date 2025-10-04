# SpiderSolver
SpiderSolver: A Geometry-Aware Transformer for Solving PDEs on Complex Geometries


# Bidirectional Projection-Based Multi-Modal Fusion Transformer for Early Detection of Cerebral Palsy in Infants

##  🧾 1.  Abstract


Periventricular white matter injury (PWMI) is the most common magnetic resonance imaging (MRI) finding in infants with cerebral palsy (CP). This work aims to detect CP and identify subtle, sparse PWMI lesions in infants under two years old with immature brain structures.

To this end, we construct a multi-modal dataset consisting of 243 cases, each with:
- Region masks of five anatomically defined target areas on T1-weighted imaging (T1WI),
- Lesion annotations on T2-weighted imaging (T2WI),
- Diagnostic labels (CP or Non-CP).

We further propose a **Bidirectional Projection-Based Multi-Modal Fusion Transformer (BiP-MFT)**, which integrates cross-modal features using a novel **Bidirectional Projection Fusion Module (BPFM)** to align anatomical regions (T1WI) with lesion patterns (T2WI).

Our BiP-MFT achieves subject-level classification accuracy of **0.90**, specificity of **0.87**, and sensitivity of **0.94**, outperforming nine competing methods by **0.10**, **0.08**, and **0.09**, respectively. Additionally, BPFM surpasses eight alternative fusion strategies based on Transformer and U-Net backbones on our dataset.

Comprehensive ablation studies demonstrate the effectiveness of the proposed annotation strategy and validate the design of the model components.


## 🧠 2.  Architecture
The two figures respectively illustrate the model's overall architecture and the design of **Bidirectional Projection Fusion Module**.

![Image text](architure2.png)
![Image text](bidirectional_module.png)



##  🏋️‍♂️ 4.  Training on the Infant-PWMl-CP Dataset

### 🔧 Training

Before training, please modify the following file paths in `BiP-MFT-2D_Infant-PWML-CP/train.py`:

- **`total_path`**: The absolute path to the `BiP-MFT-2D_Infant-PWML-CP/` directory.
- **`pretrained_weight_path`**: The path to the SegFormer weights pretrained on ImageNet-1K (`mit_b1.pth`), which can be downloaded from  
  [Google Drive](https://drive.google.com/drive/folders/1yBVICW9lcDANth-RlwJy1C9M6QNXJ0L2?usp=sharing) or  [Baidu Netdisk](https://pan.baidu.com/s/1XiwKp7Ayc81qefs3eu7pGg?pwd=fae8).

- **`data_path`**: The path to the Infant-PWML-CP dataset archive `Infant-PWML-CP.zip` (2.86 GB), downloadable from the same links above.

**Example command for training on Fold 0:**

```
CUDA_VISIBLE_DEVICES=0 python BiP-MFT-2D_Infant-PWML-CP/train.py --w1 0.2 --w2 0.5 --w3 0.1 --w4 0.2 \
  --learn_rate 0.000015 --num_epochs 30 --fold 0 --phi 'mit_b1' --batch_size 5
```


### 🧪 Evaluation

The trained model weights (`last_epoch_weights.pth`) from Fold 0 of the Infant-PWML-CP dataset are available for download:
[Google Drive](https://drive.google.com/drive/folders/1yBVICW9lcDANth-RlwJy1C9M6QNXJ0L2?usp=sharing)  or [Baidu Netdisk](https://pan.baidu.com/s/1XiwKp7Ayc81qefs3eu7pGg?pwd=fae8).

---


## 🚀 5.  



## 🛠️ 6. Requirements
The required Python packages for each code implementation are listed in their respective `requirements.txt` files.


## 📚 7. Citation
If using our Infant-PWMl-CP dataset or find this work useful in your research, please cite our paper:

```
@article{qi2025bipmft,
  title     = {Bidirectional Projection-Based Multi-Modal Fusion Transformer for Early Detection of Cerebral Palsy in Infants},
  author    = {Kai Qi and Tingting Huang and Chao Jin and Yizhe Yang and Shihui Ying and Jian Sun and Jian Yang},
  journal   = {IEEE Transactions on Medical Imaging},
  year      = {2025},
  note      = {Accepted}
}
```




## 🙏 8. Acknowledgement

We would like to acknowledge the contributions of the following works, which inspired and supported our research:

- Xie, E., Wang, W., Yu, Z., et al. **SegFormer: Simple and efficient design for semantic segmentation with transformers**. *NeurIPS*, 34 (2021), pp. 12077-12090.
- Perera, S., Navard, P., Yilmaz, A. **SegFormer3D: An Efficient Transformer for 3D Medical Image Segmentation**. *CVPR*, 2024, pp. 4981-4988.
- Lin, J., Chen, C., Xie, W., et al. **CKD-TransBTS: Clinical knowledge-driven hybrid transformer with modality-correlated cross-attention for brain tumor segmentation**. *IEEE Transactions on Medical Imaging*, 42(8), 2023, pp. 2451-2461.









