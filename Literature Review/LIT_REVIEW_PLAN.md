# Literature Review Planning
Format will be to go through a bunch of papers, and list their main points and how they relate to the project.

## Scope - What will the lit review cover?
### Aim: 
To investigate and design ways to improve the accuracy of image processing algorithms (particularly with regards to the tasks of classification and segmentation) for the case of low-dose Medical Imaging, with a particular focus on CT and X-Ray.

In particular, this will involve the application of a task-focussed denoising approach, combined with the task itself. The main idea is to adjust the loss to focus on the task over standard denoising metrics. Incorporating the N2N / Autoencoder structure is what the design will centre around.

### Literature Review
- Simulating Dose-Related noise on CT & X-Ray Images
- Deterministic Denoising Methods
- ML-based Denoising Methods
- Denoising methods that combine these two

### Argument of the Overall Paper:
1. Demonstrating and Quantifying Misalignment
    - X-Ray Paper on Misalignment of Metrics
2. Physically Accurate Noise Simulation
    - X-Ray - issues and limitations
    - CT
3. Designing a Solution
    - v1: Autoencoder & TB Chest X-Ray
    - v2: N2N & Cancer CT Scans
    - v3: Improvement & Applications to Segmentation & Other Datasets
    - Efficiency and Code implementations
4. Abstraction
    - Finding a metric or "slider" to represent this from a high-level
5. Conclusion
    - Choosing Metrics
    - Final Design
    - Assessment thereof

## Structure of the Literature Review:
Based on "The Lit Review Funnel":

__1. Intro:__ Explanation of structure and format of Lit Review

__2. Locate:__ Provide some context into field, and describe the broad theories and trends involved in this project

- Lung Functional Analysis (4 Papers):
    - Task-based ML Paradigm & Metrics
    - Dose-related Issues --> (end with Noise)

- Dose-related Noise (6 Papers):
    - Describing Noise
    - Removing Noise (Denoising)
    - Simulating Noise (X-Ray & CT)

__3. Gaps:__ Ideally bring the discussion towards a point that you identify 
gaps in research or areas that need further investigation

1. Quantification of the IQ Metric Misalignment
2. Solution that takes this into account
3. Physically-accurate Noise Simulation

__4. Targeted Gap:__ Which gap you will focus on in this project?

    Designing an architecture that works against the Metric Misalignment Problem

__5. Focus of Approach:__ With all of the Locating, gaps etc. in mind, what techniques/approaches will you use, and review them

1. Noise Simulation Methods (6 papers):
    - xcist
    - others

2. Denoising Methods (6 papers):
    - "Representational Models" (ie Fourier, Wavelet - describe the methods of denoising or ML that involve these)
        - include some papers about these models, and associated denoising methods
        - perhaps how they have been combined with CNNs

3. ML Methods (6 papers):
    - Autoencoder-type models
        - general autoencoder
        - Noise-to-noise model


__6. Selection/Details:__ Rate and compare different tools/methods, and provide some sort of selection framework

Do this for:
- Noise Simulation
- Denoising
- ML Methods
- Metrics


__7. Argue:__ Describe the relevance and benefits of filling this gap

__8. Conclude:__ Summarise the overall argument and highlights of the Lit Review

## Strategy for Building the Lit Review:
Based on the University of Kent Lit Review Guide.

__Step 1:__ Selecting the Literature - go through online libraries etc. and create a reading list of items to go over (with some idea of where it will be placed in the structure)

__Step 2:__ Skim through the main parts of each paper, and summarize/review it based on the following:
- Summarise the content (main ideas and evidence)
- Critique: Strengths and Weaknesses of the argument
- Relevance: to the chosen approach/thinking/research

__Step 3:__ Write it up, into the structure shown above

This will be done in chunks for each section of the Lit Review, which will obviously be reviewed and edited at the end.

### Schedule:

Monday: Intro + Locate

Tuesday: Locate (Finish)

Wednesday: Gap + Targeted Gaps

Thursday: Targeted Gaps (Finish) + Focus of the Approach

Friday: Focus of the Approach (finish) + Argue

Saturday - Sunday: Argue + Conclusion + Review Edits

