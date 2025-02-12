# R2A-UNET: double attention mechanisms with residual blocks for enhanced MRI image segmentation

#### Authors : Noura Bentaher · Samira Lafraxo · Younes Kabbadj · Mohamed Ben Salah · Mohamed El Ansari · Soukaina Wakrim

This study proposed a two-path encoder-decoder structure, which incorporates Normalized Channel Attention (NCA) in the contracting path with Residual Blocks
and BatchNormalization layers, these components aid in enhancing feature learning, mitigating the vanishing gradient issue, stabilizing and accelerating training, and reducing spatial
dimensions which helps capture hierarchical features at multiple scales. A key component of
the architecture is the integration of Normalized Spatial Attention (NSA) and Attention Gates
(AG) through Skip Connections, which facilitate efficient communication across layers and
maintain critical spatial information. By carefully combining the various parts, the model’s
overall performance will be improved by optimizing both feature extraction and reconstruction. This integration of advanced techniques and normalized attention mechanisms allows
the R2A-UNET to outperform existing state-of-the-art methods, it also provides good partitioning accuracy. The model’s ability to preserve spatial information relevant to tumor
localization and to focus on distinguishing between normal and abnormal tissues sets it apart
from conventional approaches. It has demonstrated advanced capabilities in medical image processing.

![image](https://github.com/user-attachments/assets/4d907bef-15d3-4b41-b698-c90c46f78690)

## Diagram of proposed method
![image](https://github.com/user-attachments/assets/acf43751-150f-4f46-ac04-469628c58363)

### NCA Block
![image](https://github.com/user-attachments/assets/6b8ef1fc-26a0-4de2-8f86-2d9d9a39c9b8)
### NSA Block
![image](https://github.com/user-attachments/assets/0a0c84b1-01d0-4dff-ba22-9ec9c9aabf3f)

## Predictions
![image](https://github.com/user-attachments/assets/6616ceda-6f8d-448c-b86a-cf0a993b0858)

## Citation 
@article{bentaher2025r2a,
  title={R2A-UNET: double attention mechanisms with residual blocks for enhanced MRI image segmentation},
  author={Bentaher, Noura and Lafraxo, Samira and Kabbadj, Younes and Ben Salah, Mohamed and El Ansari, Mohamed and Wakrim, Soukaina},
  journal={Multimedia Tools and Applications},
  pages={1--31},
  year={2025},
  publisher={Springer}
}
