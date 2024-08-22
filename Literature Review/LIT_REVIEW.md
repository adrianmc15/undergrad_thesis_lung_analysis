# Lit Review Research

    (-) poorly written / not useful (maybe an example of what not to do)
    (.) average quality/usefulness
    (+) good/useful
    (*) excellent/very useful (hero)

## Locate (16 papers):
- Lung Functional Analysis (8 Papers):
    - Task-based ML Paradigm & Metrics
        
        Relevant:
            
            1. (-) [2019, IEEE ICECCT] A Comparative Study of Lung Cancer Detection using Machine Learning Algorithms https://ieeexplore.ieee.org/abstract/document/8869001
            2. (+) [2016, IEEE Medical Imaging] Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning? (https://ieeexplore.ieee.org/document/7426826) 
            3. (*) [2021, Nature] Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans (https://www.nature.com/articles/s42256-021-00307-0)
            4. (.) [2003, Academic Radiology] Automated lung segmentation for thoracic CT: Impact on computer-aided diagnosis (https://www.sciencedirect.com/science/article/pii/S1076633204003745)
            5. (.) [2001, IEEE Transactions on Medical Imaging] Automatic lung segmentation for accurate quantitation of volumetric X-ray CT images (https://doi.org/10.1109/42.929615)
            6. (+) [2024, Medical Physics] A multiscale 3D network for lung nodule detection using flexible nodule modeling
            
            
        Extra:

            [2021, Diagnostics] Brain Hemorrhage Classification in CT Scan Images Using Minimalist Machine Learning (https://www.mdpi.com/2075-4418/11/8/1449)
            [2021, Nature] COVID-CT-MD, COVID-19 computed tomography scan dataset applicable in machine learning and deep learning https://www.nature.com/articles/s41597-021-00900-3
            [1999, ] MEDICAL DIAGNOSIS OF STROKE USING INDUCTIVE MACHINE LEARNING (https://www.researchgate.net/profile/Georgios-Dounias-2/publication/2819899_Medical_Diagnosis_Of_Stroke_Using_Inductive_Machine_Learning/links/0fcfd51407a635db88000000/Medical-Diagnosis-Of-Stroke-Using-Inductive-Machine-Learning.pdf)
            [2023, IEEE ICET] Identifying COVID-19 through X-ray and CT scan images using Machine Learning (https://ieeexplore.ieee.org/document/10375006)
            [2004, IEEE Intelligent Sensors] Automatic lung segmentation: a comparison of anatomical and machine learning approaches (https://ieeexplore.ieee.org/abstract/document/1417503)
            [2016, IEEE Medical Imaging] Lung Pattern Classification for Interstitial Lung Diseases Using a Deep Convolutional Neural Network (https://ieeexplore.ieee.org/document/7422082)
            [2022, "Genomics, Proteomics & Bioinformatics"] Machine Learning for Lung Cancer Diagnosis, Treatment, and Prognosis (https://academic.oup.com/gpb/article/20/5/850/7230459)



    - Dose-related Issues --> (end with Noise)

            7. (*) [2007, Journal of Nuclear Medicine Technology] Principles of CT: Radiation Dose and Image Quality (https://tech.snmjournals.org/content/35/4/213.full)
            8. (+) [2002, Ped Radiol] Dose and image quality in CT (https://link.springer.com/article/10.1007/s00247-002-0796-2)
            


- Dose-related Noise (8-10 Papers):
    - Describing Noise

            [2012, Review of Cardiovascular Therapy] Low-dose cardiac imaging: reducing exposure but not accuracy (https://go-gale-com.ezproxy.uct.ac.za/ps/i.do?p=AONE&u=unict&id=GALE|A283018093&v=2.1&it=r)
            [2018, Sensors] Poisson–Gaussian Noise Analysis and Estimation for Low-Dose X-ray Images in the NSCT Domain (https://www.mdpi.com/1424-8220/18/4/1019)
            [2004, AJR] Low-Dose Chest CT: Optimizing Radiation Protection for Patients https://ajronline.org/doi/full/10.2214/ajr.183.3.1830809

    - Removing Noise (Denoising)

            *[2020, Biomed. Sig. Proc. & Control] A review on medical image denoising algorithms https://www.sciencedirect.com/science/article/pii/S1746809420301920
            *[2018, Biomed. Sig. Proc. & Control] A review on CT image noise and its denoising (https://doi.org/10.1016/j.bspc.2018.01.010)
            [2017, IEEE Transactions on Medical Imaging] Robust Low-Dose CT Sinogram Preprocessing via Exploiting Noise-Generating Mechanism (https://ieeexplore.ieee.org/abstract/document/8086204)
            [2012, IPEM] Ultra-low dose CT attenuation correction for PET/CT (https://iopscience.iop.org/article/10.1088/0031-9155/57/2/309/meta)
            *[2020, CVPR] Wavelet Integrated CNNs for Noise-Robust Image Classification (https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.pdf)


    - Simulating Noise (X-Ray & CT)

            [2013, Med Phys] Realistic simulation of reduced-dose CT with noise modeling and sinogram synthesis using DICOM CT images (https://doi.org/10.1118/1.4830431)
            *[2020, Med Phys] Low-dose CT image and projection dataset (https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.14594)
            *[2022, Phys Med Biol] XCIST-an open access x-ray/CT simulation toolkit (https://pubmed.ncbi.nlm.nih.gov/36096127/)
            [2024, Phys Eng Med] Development and tuning of models for accurate simulation of CT spatial resolution using CatSim (https://pubmed.ncbi.nlm.nih.gov/38252976/)
            [2002, AJR] Computer-Simulated Radiation Dose Reduction for Abdominal Multidetector CT of Pediatric Patients https://ajronline.org/doi/full/10.2214/ajr.179.5.1791107


__3. Gaps:__ Ideally bring the discussion towards a point that you identify 
gaps in research or areas that need further investigation

1. Quantification of the IQ Metric Misalignment
2. Solution that takes this into account
3. Physically-accurate Noise Simulation

__4. Targeted Gap:__ Which gap you will focus on in this project?

    Designing an architecture that works against the Metric Misalignment Problem

__5. Focus of Approach:__ With all of the Locating, gaps etc. in mind, what techniques/approaches will you use, and review them

1. Noise Simulation Methods (ref papers above):
    - xcist
    - others

2. Denoising Methods (ref papers above):
    - "Representational Models" (ie Fourier, Wavelet - describe the methods of denoising or ML that involve these)
        - include some papers about these models, and associated denoising methods
        - perhaps how they have been combined with CNNs

3. ML Methods (4 papers):
    - Autoencoder-type models
        - general autoencoder
        - Noise-to-noise model

4. ML denoising (10 papers)

__6. Selection/Details:__ Rate and compare different tools/methods, and provide some sort of selection framework

Do this for:
- Noise Simulation
- Denoising
- ML Methods
- Metrics


__7. Argue:__ Describe the relevance and benefits of filling this gap

__8. Conclude:__ Summarise the overall argument and highlights of the Lit Review

### Template Review
#### Summary

Main ideas and evidence.

#### Critique
Strengths/Weaknesses of the presented argument.

#### Relevance
How does it link to what you are trying to do?

-----

### 1. A Comparative Study of Lung Cancer Detection using Machine Learning Algorithms 
    [2019, IEEE ICECCT] https://ieeexplore.ieee.org/abstract/document/8869001
#### Summary

Simple implementation of a few different ML algorithms (LR, Decision Trees, Naive Bayes and SVM) to a couple of Lung Cancer datasets
#### Critique
__Strengths:__ Explained the different algorithms briefly <br>
__Weaknesses:__ Written poorly, not enough information on the specifics of implementation, dataset was not described in enough detail, only used accuracy to measure the effectiveness/value of the different approaches, and so lacked some critical analysis

#### Relevance
Possibly an indication of trends in ML and Cancer classification. Maybe useful for a quick survey of diff. ML algo's. Not a very good paper

-----

### 2. Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning? 
    [2016, IEEE Medical Imaging] https://ieeexplore.ieee.org/document/7426826

#### Summary

Essentially focussed on comparing training CNN's from scratch or using pre-trained models and using shallow or deep fine-tuning on the particular dataset. 4 different datasets & imaging modalities were provided, that were meant to represent the most common tasks/applications of this kind of paradigm to medical imaging, one of which was PE detection from CT scans.

The results were essentially that, in general pre-trained models perform better or equally as well as those trained from scratch. This was evident in the PE example (where performance was relatively equal), along with the fact that when the size of the training set decreases, the pre-trained models outperform the from-scratch approach. Additionally, it was found that "deep fine-tuning" is better than "shallow fine-tuning" every time.

#### Critique
__Strengths:__
- definitely scientifically sound (the analysis was deep and considered a few different aspects)
- broad and rigorous analysis
- provides a detailed background and rich related work

__Weaknesses:__
- slightly outdated - using AlexNet, which is a bit behind the more modern U-nets
- they did not fully optimise their hyperparameters in the CNNs, which slightly weakens the design of their networks

#### Relevance
- Demonstrates the modern interest in using ML for tasks etc. (around the beginning of that revitalised interest)
- The pre-trained model might be useful for what we are trying to do (and is common in modern applications)
-----

### 3. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans 
    [2021, Nature] (https://www.nature.com/articles/s42256-021-00307-0)

#### Summary
A review of many papers that were published about detection of COVID-19 using Chest radiographs and CT scans. The paper essentially outlines important things to be considered when writing an ML paper in the medical imaging field and demonstrates how few ML papers actually achieve these important characteristics. The result was ultimately that 0 papers during the time from Jan to Oct of 2020 could be considered applicable for implementation in a clinical setting, but the paper also outlines and provides a framework for setting up scientifically sound methodology for medical imaging and ML.

#### Critique
__Strengths:__ Very well-written, systematic and clear at describing "what makes a good paper" <br>
__Weaknesses:__ focussed on COVID-19, while we are focussed on CT 

#### Relevance
- sets up a collection of consideraiions to take into account when developing ML paper in this field
- demosntrating popularity - and importance of ML in Lung Functional Analysis is so important.

---

Potentially useful for comparing "deterministic" and ML approaches, but excluded for now:
    
    [2004, IEEE Intelligent Sensors] Automatic lung segmentation: a comparison of anatomical and machine learning approaches (https://ieeexplore.ieee.org/abstract/document/1417503)
            
---

### 4. Automated lung segmentation for thoracic CT: Impact on computer-aided diagnosis
    [2003, Academic Radiology] (https://www.sciencedirect.com/science/article/pii/S1076633204003745)

#### Summary
Describes a deterministic (non-ML) approach to thoracic lung segmentation, for the purpose of lung nodule classification and the measurement of tumor thickness in mesothelioma. The technique described was a modification from an existing one, so focussed on measuring how it improved from the original version. The paper is quite old so still using simple techniques without "data-driven" approaches. The results were relatively good, but limited. Lung nodule classification done failed 4.9% of the time which was an improvement from 14% without the modifications. A correlation coefficient of 0.990 was reported for mesothelioma, which was an improvement form 0.977.

#### Critique
__Strengths:__ very descriptive, and had good statistical analysis, very clear process and likely reproducable, seemed quite successful considering that there is no ML involved <br>
__Weaknesses:__ older paper so no ML or datadriven stuff, very limited size of dataset (only 38)

#### Relevance
Demonstrates the importance of segmentation. An example of classical "deterministic" techniques, which could be useful to think about. Perhaps a segmentation step like this would be useful in this thesis.

---
### 5. Automatic lung segmentation for accurate quantitation of volumetric X-ray CT images
    [2001, IEEE Transactions on Medical Imaging] (https://doi.org/10.1109/42.929615)

#### Summary
Provides a fundamental method for automatic lung segmentation, with the aim of creating a method optimal for 3D CT modelling. A few improvements were made, including dynamic thresholding in the first step (as opposed to using a single threshold), an efficient method to find the anterior and posterior junction lines between the right and left lungs, and optionally smoothing the irregular boundary at the mediastinum. The results were good with an rms value of 8 pixels off the human segmented versions

#### Critique
__Strengths:__ Lots of statistical analysis, and detail provided about the methodology (and math equations involved) <br>
__Weaknesses:__ ...

#### Relevance
Demonstrates the importance of segmentation. An example of classical "deterministic" techniques, which could be useful to think about. Perhaps a segmentation step like this would be useful in this thesis.

---

Might be useful to look at, if looking for more lung cancer related classification based on CT:
        
    [2022, Med. Phys.] Differentiation between immune checkpoint inhibitor-related and radiation pneumonitis in lung cancer by CT radiomics and machine learning (https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15451) 

---
### 6. A multiscale 3D network for lung nodule detection using flexible nodule modeling
    [2024, Medical Physics] (https://doi.org/10.1002/mp.17283)

#### Summary
They used a 3D-type model for lung nodule detection. The architecture seems quite modern and well-structured, similar to a U-Net. Their results were 90% accuracy which is apparently an improvement from previous work. The paper includes an excellent discussion on other methods, including pre-trained deep-learning methods (such as YOLO and R-CNN). 
#### Critique
__Strengths:__ excellent statistical analysis (ablation studies, detailed explanations of methods and analysis of results), nice relevant designed architecture (modern and makes sense for the purpose), really nice dataset size (LUNA 16) <br>
__Weaknesses:__ not sure

#### Relevance
A similar task to thesis plans. Similar network structure to U-net idea. Very useful discussion on modern classification/segmentation methods, in particular with pre-trained models and deep learning in mind.

### 7. Principles of CT: Radiation Dose and Image Quality 
    [2007, Journal of Nuclear Medicine Technology] (https://tech.snmjournals.org/content/35/4/213.full)

#### Summary
Essentially, an explanation of 3 essential issues in CT dose and image quality: 1) CT dose and the measurement of it 2) CT image quality. The chapter essentially explains how CT image dose works, and how CT scans actually cause issues with image quality from a physical perspective. It also goes through the main ways of measuring image quality and dose and the respective pitfalls

#### Critique
__Strengths:__ Informative and useful background information, describes several different aspects
__Weaknesses:__ No actual experiment was done, so there are no results or arguments to refute etc., does not mention CNR

#### Relevance
It basically provides a background explanation of how dose and image quality are related, which are important for the physically accurate simulation of noise and dose-related image quality reduction. Also useful for understanding how CT works, which is important for modelling it accurately.

---

### 8. Dose and image quality in CT 
    [2002, Ped Radiol] (https://link.springer.com/article/10.1007/s00247-002-0796-2)

#### Summary
The paper explains the different relationships between CT scanner settings and the resulting dose measurements on patients. A primary takeaway is the difference between CTDI, Effective Dose and Radiation Energy transfer, and their relationship to the risk for the patient. The paper also describes exactly how the radiation doses actually affect the patient medically. Mentioned as well are several metrics used to describe CT image quality.

#### Critique
__Strengths:__ Clear explanation, informative. Critical analysis on metrics <br>
__Weaknesses:__ Limited in scope to Pediatrics, but still relevant in terms of the information provided.

#### Relevance
Provides a good starting point on relevant metrics to use in the study in terms of dose, but also provides the basis of how image quality and noise is described and quantified in CT imaging.