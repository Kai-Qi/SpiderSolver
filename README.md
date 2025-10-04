# SpiderSolver
SpiderSolver: A Geometry-Aware Transformer for Solving PDEs on Complex Geometries


# SpiderSolver: A Geometry-Aware Transformer for Solving PDEs on Complex Geometries，NeurIPS 2025.

##  🧾 1.  Abstract

 Transformers have demonstrated effectiveness in solving partial differential equa
tions (PDEs). However, extending them to solve PDEs on complex geometries
 remains a challenge. In this work, we propose SpiderSolver, a geometry-aware
 transformer that introduces spiderweb tokenization for handling complex domain
 geometry and irregularly discretized points. Our method partitions the irregular
 spatial domain into spiderweb-like patches, guided by the domain boundary ge
ometry. SpiderSolver leverages a coarse-grained attention mechanism to capture
 global interactions across spiderweb tokens and a fine-grained attention mechanism
 to refine feature interactions between the domain boundary and its neighboring
 interior points. We evaluate SpiderSolver on PDEs with diverse domain geometries
 across five datasets, including cars, airfoils, blood flow in the human thoracic aorta,
 as well as canonical cases governed by the Navier-Stokes and Darcy flow equations.
 Experimental results demonstrate that SpiderSolver consistently achieves state-of-the-art performance across different datasets and metrics, with better generalization
 ability in the OOD setting.


## 🧠 2.  Architecture
The two figures respectively illustrate the model's overall architecture and the design of **Bidirectional Projection Fusion Module**.

![Image text](architecture.png)




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









