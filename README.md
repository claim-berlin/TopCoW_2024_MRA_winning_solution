# TopCoW 2024 Solution - Charité Lab for AI in Medicine

## Overview
Inference repository for the winning solution in the MICCAI 2024 - TopCoW Challenge (TOF-MRA track)

- Challenge Website: https://topcow24.grand-challenge.org/  
- [Leaderboard](https://topcow24.grand-challenge.org/evaluation/finaltest-mra-task-1-seg/leaderboard/)

Developed by the Charité Lab for AI in Medicine (CLAIM) research group at Charité University Hospital, Berlin  
Team working on this solution: 
Orhun Utku Aydin, Jana Rieger, Adam Hilbert, Dimitrios Rallios, Satoru Tanioka, Dietmar Frey

### Please cite the TopCoW paper and papers listed below in the Dataset and references sections if you use this model in your research

## Model 
1. **Input:**
   - Time-of-Flight Magnetic Resonance Angiography (TOF-MRA) or Computed Tomography Angiography (CTA) of the brain without any image preprocessing

2. **Output:**
   - Multiclass segmentation of the Circle of Willis artery segments:  
     "background": 0,  
     "BA": 1,  
     "R-PCA": 2,  
     "L-PCA": 3,  
     "R-ICA": 4,  
     "R-MCA": 5,  
     "L-ICA": 6,  
     "L-MCA": 7,  
     "R-Pcom": 8,  
     "L-Pcom": 9,  
     "Acom": 10,  
     "R-ACA": 11,  
     "L-ACA": 12,  
     "3rd-A2": 13  

## Inference

To set up the environment and install the necessary dependencies, follow these steps:

#### Build conda environment
1. Create and Activate a Virtual Environment  
```bash
conda create -n topcow_claim python==3.11   
conda activate topcow_claim  
 ```

2. Install the requirements
```bash
pip install ultralytics
cd topcow-2024-nnunet
pip install -e .
```

#### Download model weights
Download model weights from Zenodo:  
- https://zenodo.org/records/14191592

Place models inside **models** folder:
- models/yolo-cow-detection.pt
- models/topcow-claim-models

### Running inference

Run the segment_circle_of_willis.py specifying following variables in the beginning of the segment_circle_of_willis.py script:  
(1) YOLO_FILE_PATH: the path to the yolo model    
(2) SEGMODEL_DIR_PATH: the folder containing the trained nnUnet segmentation models  
(3) IMG_FOLDER: the folder containing the input CTA or TOF-MRA images  
(4) SEGMENTATION_FOLDER: the folder where the segmentation results should be stored   

## Training details 
An additional 500 patients (250 CTA, 250 MRA) were included in the training dataset. All used in-house and public datasets are detailed in the Table below.  
Inclusion criteria on additional data were:   
(1) No intracranial aneurisms in CoW ROI **AND**   
(2) Baseline segmentation model made topological mistakes **OR**  
(3) Rare variants in 3D vessel geometry such as 3rd A2 segment, hypoplastic A1 segment, patients with vessel occlusions 

First, TopCoW data was used to create the baseline, pre-labeling model. We used this model to create labels for a large (>3000) set of scans, that were used to 1) pre-select additional patients following above inclusion criteria and 2) to manually refine segmentation labels for ground truth. 
Second, once labels were refined we used 125-125 MRA and CTA scans additionally to TopCoW data for internal model experiments, review and hyper parameter tuning.
The full dataset used to train our final model for submission contained 750 scans (TopCoW+500 additional). 

Manual segmentation was based on initial pre-labeling from the baseline model and refined by 1 junior rater (1-2 years experience with vessel segmentation) and 1 senior rater (5+ years experience with vessel segmentation). Special attention was paid to correcting Acom, Pcom, 3rd A2 and unusual geometries. Manual segmentations were done in ITK-Snap.

2. **Model training:**
   - **ResEncM** model of the nnUNet framework
   - trained using 5-fold cross-validation 
   - Skeleton-Recall loss utilized  (see References)

### Training Dataset
| Dataset                       | Modality | Type      | Included Patients | Details 
|-------------------------------|----------|-----------|-------------------|------------- |
| Japan Intracranial Hemorrhage | CTA      | private   | 53                | CTAs of patients with spontaneous ICH from 4 Japanese hospitals  |
| MRCLEAN<sup>1</sup>           | CTA      | private   | 109               | CTAs from the MR CLEAN trial data, ischemic stroke   |
| IA Large Aneurysm<sup>2</sup> | CTA      | public    | 88                | Includes 1000+ CTA images, multi-center, aneurysm  |
| OASIS<sup>3</sup>             | TOF-MRA  | public    | 6                 | Neurodegenerative disease, single center   |
| NITRC<sup>4</sup>             | TOF-MRA  | public    | 1                 | Healthy patients, single center  |
| LAUSANNE<sup>5</sup>          | TOF-MRA  | public    | 23                | Healthy patients, single center   |
| IXI<sup>6</sup>               | TOF-MRA  | public    | 44                | Healthy patients, multi-center  |
| ICBM<sup>7</sup>              | TOF-MRA  | public    | 32                | Healthy patients, multi-center |
| CASILAB<sup>8</sup>           | TOF-MRA  | public    | 55                | Healthy patients, single center   |
| 1kplus<sup>9</sup>            | TOF-MRA  | private   | 78                | Ischemic stroke patients, with or without LVO, single center, Charité Berlin   |
| EDEN<sup>10</sup>             | TOF-MRA  | public    | 11                | Healthy patients, multi-center   |

1 Berkhemer et al. “A Randomized Trial of Intraarterial Treatment for Acute Ischemic Stroke”. N Engl J Med. 2015;372(1):11-20. 

2 Bo, Z.-H. “Large IA Segmentation dataset”. Zenodo, 2021 

3 LaMontagne et al., “OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease” (preprint), 2019, Radiology and Imaging. https://doi.org/10.1101/2019.12.13.19014902 

4 NITRC: Magnetic Resonance Angiography Atlas Dataset: Tool/Resource Info [WWW Document], n.d. URL https://www.nitrc.org/projects/icbmmra/ (accessed 3.7.24). 

5 Di Noto et al., “Towards Automated Brain Aneurysm Detection in TOF-MRA: Open Data, Weak Labels, and Anatomical Knowledge.”, 2023, Neuroinformatics 21, 21–34. https://doi.org/10.1007/s12021-022-09597-0 

6 IXI Dataset – Brain Development, n.d. URL https://brain-development.org/ixi-dataset/ (accessed 3.7.24). 

7 Mazziotta, J et al., “A probabilistic atlas and reference system for the human brain: International Consortium for Brain Mapping (ICBM).”, 2001, Philos. Trans. R. Soc. Lond. Ser. B 356, 1293–1322. https://doi.org/10.1098/rstb.2001.0915 

8 Bullitt, E. et al., “Vessel tortuosity and brain tumor malignancy: a blinded study.”, 2005, Acad. Radiol. 12, 1232–1240. https://doi.org/10.1016/j.acra.2005.05.027 

9 Hotter B et al., “Prospective study on the mismatch concept in acute stroke patients within the first 24 h after symptom onset - 1000Plus study.” BMC Neurol. 2009 Dec 8;9:60. doi: 10.1186/1471-2377-9-60. PMID: 19995432; PMCID: PMC3224745. 

10 Castellano, A., et al. “EDEN2020 Human Brain MRI Datasets for Healthy Volunteers (1.0)”, Zenodo, 2019 


### References
- Yang, K., Musio, F., Ma, Y., Juchler, ... , Menze, B., 2024. TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA. https://doi.org/10.48550/arXiv.2312.17670
- Kirchhoff, Y., Rokuss, M.R., Roy, S., Kovacs, B., Ulrich, C., Wald, T., Zenk, M., Vollmuth, P., Kleesiek, J., Isensee, F., Maier-Hein, K., 2024. Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures, in: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., Varol, G. (Eds.), Computer Vision – ECCV 2024. Springer Nature Switzerland, Cham, pp. 218–234. https://doi.org/10.1007/978-3-031-72980-5_13   
- Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z



