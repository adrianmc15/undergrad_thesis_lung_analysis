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
            9. (+) [2021, IEEE Access] U-Net and Its Variants for Medical Image Segmentation: A Review of Theory and Applications (https://doi.org/10.1109/ACCESS.2021.3086020)
            
            
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

            10. (*) [2020, Biomed. Sig. Proc. & Control] A review on medical image denoising algorithms https://www.sciencedirect.com/science/article/pii/S1746809420301920
            11. (*) [2018, Biomed. Sig. Proc. & Control] A review on CT image noise and its denoising (https://doi.org/10.1016/j.bspc.2018.01.010)
            12. (*) [2023, Medical Physics] CT image denoising methods for image quality improvement and radiation dose reduction https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/acm2.14270
            13. (+) [2020, CVPR] Wavelet Integrated CNNs for Noise-Robust Image Classification (https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.pdf)

        ^These reviews are very useful, so include references to some of the best algorithms described (maybe 4 references)^ 
        Included below

            14. (.) D.-H. Trinh, T.-T. Nguyen, N. Linh-Trung, An effective example-based denoising method for CT images using Markov random field, in: Proc. IEEE Int. Conf. Advanced Technologies for Communications (ATC 2014), IEEE, Hanoi, 2014, pp. 355–359.

            15. L. A. Shepp and B. F. Logan, "The Fourier reconstruction of a head section," in IEEE Transactions on Nuclear Science, vol. 21, no. 3, pp. 21-43, June 1974, doi: 10.1109/TNS.1974.6499235. keywords: {Interpolation;Search methods;Fourier transforms;Bandwidth;Oscillators;Approximation algorithms;Spatial resolution},

            16. Mohammadinejad P, Mileto A, Yu L, et al. CT noise-reduction methods for lower-dose scanning: strengths and weaknesses of iterative reconstruction algorithms and new techniques. RadioGraphics. 2021;41(5):1493-1508.

            17. Antoni Buades, Bartomeu Coll, Jean-Michel Morel. A review of image denoising algorithms, with a new one. Multiscale Modeling and Simulation: A SIAM Interdisciplinary Journal, 2005, 4 (2), pp.490-530. ff10.1137/040616024ff. ffhal-00271141f

            18. L. I. Rudin, S. Osher, and E. Fatemi, Nonlinear total variation based noise removal algorithms, Physica D, 60 (1992), pp. 259–268.

            19. Mallat, S.G. A Theory for Multiresolution Signal Decomposition: The Wavelet Representation (1989) IEEE Transactions on Pattern Analysis and Machine Intelligence, 11 (7), pp. 674-693

            20. 15. Zhao T, Hoffman J, Mcnitt-Gray M, Ruan D. Ultra-low-dose CT image denoising using modified BM3D scheme tailored to data statistics. Med Phys. 2019;46(1):190-198.

            21.  H. Chen, Y. Zhang, M.K. Kalra, F. Lin, P. Liao, J. Zhou, G. Wang, Low-Dose CT with a Residual Encoder–Decoder Convolutional Neural Network (RED-CNN), 2017 arXiv preprint arXiv:1702.00288.


        Extra reading

            [2017, IEEE Transactions on Medical Imaging] Robust Low-Dose CT Sinogram Preprocessing via Exploiting Noise-Generating Mechanism (https://ieeexplore.ieee.org/abstract/document/8086204)
            [2012, IPEM] Ultra-low dose CT attenuation correction for PET/CT (https://iopscience.iop.org/article/10.1088/0031-9155/57/2/309/meta)
            

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
__Strengths:__ Informative and useful background information, describes several different aspects <br>
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

---
### 9. (+) [2021, IEEE Access] U-Net and Its Variants for Medical Image Segmentation: A Review of Theory and Applications (https://doi.org/10.1109/ACCESS.2021.3086020)

----


### 10. A review on medical image denoising algorithms
    [2020, Biomed. Sig. Proc. & Control]  https://www.sciencedirect.com/science/article/pii/S1746809420301920

#### Summary
The paper describes the following: Image Noise and how it comes about, surveys the various types of denoising methods available and goes through various metrics commonly used for describing results. It then completes a brief experiment demonstrating these things. The paper talks about a variety of modalities, but also talks a lot about CT imaging. It talks about how CT works (with sinograms and back-projection), and how low-dose noise comes about. It describes the origin of the noise (through statistical fluctuations of X-ray quanta --> quantum noise) and the modelling of it through the Poisson-Gaussian distrbution. Other models exist that use the Central Limit Theorem, that assume "Additive Gaussian statistics". The paper describes the features of a good denoising algorithm (preservation of edges, maintaining structural similarity, complexity and non-essentialness of prior databases).

It then categorizes and describes various different types of denoising algorithms for CT as being Sinogram filtering (applying filters directly in the sinogram domain), Iterative reconstruction techniques (using other information about the scan - from the DICOM header for example), and Post-processing techniques (Including the use of neural networks and image domain techniques).

![alt text](<10_table_of_methods.png>)

The paper discusses different important metrics used to describe medical image quality and noise. The main metrics used for this paper are PSNR, FSIM (Feature Similarity) and EPI (Edge Preservation Index). This is a very useful because (as the paper says), edge preservation is very important for diagnosis and segmentation. 

The brief experiment with some of the different techniques show how the methods work differently based for different metrics (none of which are task based). One experiment was done with a simulation of Gaussian Additive noise applied to a dataset of clean images. The same algorithms were then run on real LDCT images, but instead information entropy (randomness) was used as a metric of measuring the noise.

#### Critique
__Strengths:__ very good overview of what different techniques exist and how they are used <br>
__Weaknesses:__ Does not describe task-based metrics (more focussed on the denoising itself)

#### Relevance
Useful for creating an understanding of the different standard denoising techniques available. Great overview of how the noise is created and usually modelled. Great discussion of what actually useful metrics are, and also a very useful framework for categorizing and understanding the different types of denoising algorithms.

---
### 11. A review on CT image noise and its denoising
    [2018, Biomed. Sig. Proc. & Control] (https://doi.org/10.1016/j.bspc.2018.01.010)


#### Summary
A great overview of CT noise in general. The paper goes through a variety of aspects related to noise including the issues with CT image reconstruction, CT noise, different types of denoising methods and finally completes a comparative analysis of some of the main algorithms described.

Noise is difficult to model in CT because the noise distribution of the final projection / result will differ based on the reconstruction algorithm used, and various factors related to the settings and initial processing of the original machine. 

![alt text](<11_additive_gaussian_example.png>)

> "The problem to identify noise in CT image is that all the intermediate steps, like interpolations or filtering with the convolution kernel, introduce correlations to the noisy data. Due to these dependencies, the noise distribution in the final CT image is usually unknown."

There are different approaches to accurately modelling it depending on the type of scanner and the parameters. MDCT (multi-detector) is accurately described by a Gaussian distribution, while single detector (unclear) is best modeled with Poisson. But generally, based on the literature, it is additive Gaussian noise. In the comparative analysis provided by the paper, the Central Limit Theorem is used and puts forward that noise can be modeled using a Gaussian distribution. The reason for this is that each voxel in an image is the result of many different projections with differing noise distributions. The paper describes one method in particular that quite well models this fact in its denoising which would be useful to mention

Different sources of noise are described: Random noise, Statistical noise, Electronic noise and Roundoff errors, with Statistical (quantum) noise as the most important (dose-related).

Four criteria are provided for describing and comparing denoising algorithms: <br>
(i) visibility of the artifacts; <br>
(ii) preservation of edge details; <br>
(iii) visibility of low contrast objects; <br>
(iv) preservation of the texture. <br>

These correspond to the metrics discussed in the review: <br>
(i) SSIM - structural similarity <br>
(ii) RMSE <br>
(iii) PSNR <br>
(iv) IQI (Image Quality Index) <br>
(v) ED (Entropy Difference) <br>
(vi) DIV (Difference in Variance) <br>
(vii) GMSD - describes pixelwise gradient similarity <br>

The paper then goes through the descriptions of many different types of denoising algorithms in both the spatial and trasnform domain. Initially they just used LPFs, then moved onto using projection-based techniques because the reconstruction algorithms of CT tended to cause blurriness - this is where the development of filtered backprojection (FBP) comes from. It also describes Iterative Reconstruction Approaches for noise suppression which worked by optimizing statistical objective functions.

Then described are post-processing approaches in 1) the spatial domain and 2) the transform domain (mostly using the wavelet transform). Linear and non-linear filters are described, with one of the most important advances being the Wiener filter. Total Variation (TV) was another significant advancement, which led to one of the better performing algorithms used in the comparative analysis. The Split Bregman method was also significant. Dictionary Learning methods are described, which essentially uses dual energy CT info to provided better noise suppression. Bilateral and NLM (also mentioned in [10]) were also significant. This has since developed into more advanced methods combined with TV in Probabilistic NLTV (PNLTV) and Pointwise Box-Counting Dimension (PWBCD). Deep-learning algorithms such as the Autoencoder and Residual encoder decoder CNN (RED-CNN). Transform domain filtering is also described, which mostly focusses on Wavelet Transform based techniques, and the development of techniques within it, such as dynamic thresholding. BM3D, which is the state of the art for Additive White Gaussian Noise is also described.

The main algorithms are also described and tested in a comparative study, where each one is used on the same dataset. The results are in the paper, but the advantages and disadvantages are summarised below:

![alt text](<11_denoising_algo_summary.png>)

#### Critique
__Strengths:__ Overall very useful overview, will be used as a basis for the discussion of noise and denoising, will also mention papers and techniques for additional references <br>
__Weaknesses:__ ...


#### Relevance
Will be used as a basis for the discussion of noise.

---

### 12. CT image denoising methods for image quality improvement and radiation dose reduction
    [2023, Medical Physics] https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/acm2.14270

#### Summary
Provides an review of image denoising for CT (covering about 222 publications), but with a focus on the more "modern" or Deep-Learning (DL) based methods. It also discusses denoising metrics and issues of training, validation and evaluation.

In terms of Denoising Algorithms the paper provides two general categories: Traditional Noise Reduction Methods and DL-Based methods, where the former includes all of those that are not in the latter. 

It briefly describes the main Traditional Noise Reduction paradigms (and how they work) including: 
- Filtered back projection (FBP)
- Iterative reconstruction (IR)
- Wavelet-based denoising
- Non-local means (NLM)
- Total Variation (TV)
- Dictionary learning-based denoising
- Block-matching and 3D filtering (BM3D)

DL-based approaches are described in more detail (CNNs, GANs, Transformers and Others). The mechanics of CNNs are described first, with their advantages and disadvantages. RED-CNN and U-NET are described in more detail, as they are the two most popular models in the studies reviewed. Then, GANs are described, with a focus on CycleGAN and WGAN. A major advantage is their ability to work successfully even with unpaired data. Transformers are described, with their ability to capture spatial dependencies and the self-attention mechanism enabling them to non-linearly record relationships between segments of the input sequence, however require a significant amount of labeled data and so have not been substantially adopted. Other methods include VAEs, ResNets and Attention-based Networks.

The paper then describes different Training, Validation and Evaluation methods, and issues surrounding them. It describes Supervised, Unsupervised, Self-supervised and Weakly supervised methods. Particularly interesting was discussion on unsupervised-type of models - since there is little publicly available data, this would be very useful/interesting to look further into. Another useful point was the need for "independent test", which is basically a test to ensure that the model is generalizable and works on different datasets. It was only used by a few of the papers reviewed in this paper, but is apparently very important.

Metrics are discussed, including DI (Dunn's Index), IQR, SNR, PSNR, CNR, MSE, NPS, CCC (Concordance Correlation Coefficient) and SSIM are discussed. An interesting point made was that SSIM has shown high correlation with radiologists' evaluations of diagnostic quality and low-contrast detectability and moderate corr. for texture. The paper argues that the combination of these metrics provide a good overall idea of how well the denoiser has done, but they do not "perfectly \[mirror\] human visual perception", and that the selection of appropriate metrics depends on the nuances of the image denoising task at hand.

Also very useful is the list of datasets provided:

![alt text](<12_table_of_datasets.png>)

Other tools mentioned were transfer learning and data augmentation techniques. Additionally the problem of adverserial attacks was discussed.



#### Critique
__Strengths:__ <br>
__Weaknesses:__ 

#### Relevance
...

---

---

### 13. Wavelet Integrated CNNs for Noise-Robust Image Classification
    [2020, CVPR]  (https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.pdf)

#### Summary
...

#### Critique
__Strengths:__ Clear explanation, informative. Critical analysis on metrics <br>
__Weaknesses:__ Limited in scope to Pediatrics, but still relevant in terms of the information provided.

#### Relevance
...

---




DL(2)
-	GANs
-	Transformers
-	Others (Dictionary learning, VAEs)
-	Supervised vs unsupervised
Metrics
-	How to think about them:
o	They are useful for getting a good overall idea
o	Limitations
-	List and describe them (make a table)
Datasets
-	Independence Test
-	CTIA, Mayo etc.
Simulation
-	XCIST
-	Other papers
