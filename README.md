
<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
            <img src="docs/1748968628246.png" alt="Icon" style="width:40px; vertical-align:middle; margin-right:10px;">      M3C2

<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
              Joint Modelling Histology and Molecular Markers for Glioma Classification</h1>    
              
<p align='center' style="text-align:center;font-size:1.25em;">
    Xiaofei Wang<sup>a,1</sup>, 
    Hanyu Liu<sup>b,1</sup>, 
    Yupei Zhang<sup>a</sup>, 
    Boyang Zhao<sup>b</sup>, 
    Hao Duan<sup>c</sup>, 
    Wanming Hu<sup>d</sup>,
    Yonggao Mou<sup>c</sup>, 
    Stephen Price<sup>a</sup>, 
    Chao Li<sup>a,b,e,f</sup>&nbsp;<br/>
    <sup>a</sup> <strong>Department of Clinical Neurosciences, University of Cambridge, UK</strong><br/>
    <sup>b</sup> <strong>School of Science and Engineering, University of Dundee, UK</strong><br/>
    <sup>c</sup> <strong>Department of Neurosurgery, State Key Laboratory of Oncology in South China, Guangdong Provincial Clinical Research Center for Cancer, Sun Yat-sen
 University Cancer Center, China</strong><br/>
    <sup>d</sup> <strong>Department of Pathology, State Key Laboratory of Oncology in South China, Guangdong Provincial Clinical Research Center for Cancer, Sun Yat-sen University
 Cancer Center, China</strong><br/>
    <sup>e</sup> <strong>Department of Applied Mathematics and Theoretical Physics, University of Cambridge, UK</strong>
    <sup>f</sup> <strong>School of Medicine, University of Dundee, UK</strong>
</p>



<div align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S1361841525000532" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
</div>
</p>

## 📣 Latest Updates

- **[2025-03-25]** 📊 *Code have been released!*
- **[2025-02-04]** 📝 *Hypo3D paper preprint is now available on [Medical Image Analysis]([https://arxiv.org/abs/your-link](https://www.sciencedirect.com/science/article/pii/S1361841525000532)).*
- **[2024-12-04]** 🎉 *Hypo3D has been accepted to Medical Image Analysis!*
- **[2024-08-02]** 📝 *M3C2 are now submitted to the Medical Image Analysis.*

## Key Takeaways

- **M3C2** presents a groundbreaking framework for cancer classification by integrating histology and molecular markers.  
  🧠 **Key Innovation**: The framework employs **multi-scale disentangling modules** to extract both **high-magnification cellular-level** and **low-magnification tissue-level** features, which are then used to predict histology and molecular markers simultaneously.

- The method introduces a **Co-occurrence Probability-based Label-Correlation Graph (CPLC-Graph)** to model the relationships between multiple molecular markers.  
  This enhancement leads to better classification accuracy by capturing **intrinsic marker co-occurrences** and their impact on cancer classification.

- **Cross-Modal Interaction** is key to the model’s success.  
  🔄 **Interaction Mechanism**: By using **dynamic confidence constraints** and a **cross-modal gradient modulation strategy**, M3C2 efficiently aligns the prediction tasks for histology and molecular markers, ensuring both tasks complement each other for more accurate results.

- **Validation Across Diverse Datasets**: M3C2 outperforms existing state-of-the-art methods in **glioma classification** and **molecular marker prediction**, showcasing its robustness in **internal** and **external validation datasets**.  
  📊 **Performance**: The method achieves significant improvements, with **accuracy** and **AUC scores** surpassing previous models by as much as **5.6%** in certain tasks.

- **Clinical Implications**: The ability to predict molecular markers directly from **whole-slide images (WSIs)**, combined with the model's capacity to understand the interactions between histology and molecular data, offers strong potential for **precision oncology**.  
  🏥 **Impact**: M3C2’s approach aligns with the latest **WHO glioma classification criteria**, making it a promising tool for clinical decision-making and personalized cancer treatment.


![sicl](docs/framework图.png)

## About this code


## How to apply the work
### 1. Environment
- Python >= 3.7
- Pytorch >= 1.12 is recommended
- opencv-python
- sklearn
- matplotlib


### 2. Train
Use the below command to train the model on our database.
```
    python ./main.py 
```

### 3. Test
Use the below command to test the model on our database.
```
    python ./test_main.py
```

### 4. Datasets
```
https://www.kaggle.com/datasets/liuhanyu1007/m3c2-data
```

### 5. Model
```
https://www.kaggle.com/models/liuhanyu1007/m3c2_model
```

## Contact
- Hanyu Liu: 2485644@dundee.ac.uk

Please open an issue or submit a pull request for issues, or contributions.

## 💼 License

<a href="https://opensource.org/licenses/MIT" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
</a>

## Citation

If you find our benchmark is helpful, please cite our paper:

```
@article{wang2025joint,
  title={Joint Modelling Histology and Molecular Markers for Cancer Classification},
  author={Wang, Xiaofei and Liu, Hanyu and Zhang, Yupei and Zhao, Boyang and Duan, Hao and Hu, Wanming and Mou, Yonggao and Price, Stephen and Li, Chao},
  journal={arXiv preprint arXiv:2502.07979},
  year={2025}
}
```
