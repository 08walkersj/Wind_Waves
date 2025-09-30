# Auroral Kilometric Radiation Measured by WIND
This repository focuses on tools to aid research of auroral kilometric radiation using the RAD1 instrument that is part of the WAVES experiment on board WIND.
## This Repository Includes:
- Tools for reading the RAD1 raw data and creating flag columns for data quality and usability
- Tools for plotting spectragrams, implementing gap filling in frequency and time and filtering out non AKR sources
- Tools for plotting the distribution of AKR frequency limits
- Tools for calculating integrated power

## Useful Examples
high_resolution_spectragram.ipynb in the notebooks folder demonstrates a working example of the spectragram, gap filling and AKR seletion techniques. 

## Relevant Papers
This repository was developed to provide the code and methodology behind the following papers:
- Walker 2025
The notebooks folder not only contains examples to help users with this repository but the methodology behind the figures from each paper and a demonstration of how to recreate the figures



## Walker 2025
For Walker 2025 to recreate the work in the study the following flow chart should be followed:
```mermaid
flowchart TD
    A[Step 1: Download_Data.ipynb] --> B[Step 2: Defining_Substorm_Epoch.ipynb]
    B --> C[Step 3: Calculating_Frequency_Extension.ipynb]
    C --> D[Step 4: Calculating_Integrated_Power.ipynb]
    D --> E[Step 5: Figure Notebooks in Any Order]


