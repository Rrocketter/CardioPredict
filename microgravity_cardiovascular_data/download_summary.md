# Microgravity Cardiovascular Data Download Summary

## Project Overview
This collection supports the development of an ML model to predict cardiovascular risk in astronauts and bedridden patients based on physiological markers from microgravity exposure.

## Download Results

### Successfully Downloaded (7 datasets):
- **OSD-575**: SpaceX Inspiration4 Blood Serum Metabolic Panel and Immune/Cardiac Cytokine Arrays
- **GSE14798**: Woman skeletal muscle transcriptome with bed rest and countermeasures
- **OSD-51**: Woman skeletal muscle transcriptome with bed rest (OSD version)
- **OSD-635**: Spaceflight effects on human vascular smooth muscle cell phenotype and function
- **OSD-484**: Astronauts plasma-derived exosomes induced gene expression in AC16 cells
- **GSE137081**: Effects of Spaceflight on Human iPSC-Derived Cardiomyocytes
- **OSD-258**: Effects of Spaceflight on Human iPSC-Derived Cardiomyocytes (OSD version)

### Failed Downloads (0 datasets):

## Next Steps

1. **Data Exploration**: Start with OSD-575 (Inspiration4 biomarkers) as your anchor dataset
2. **Feature Engineering**: Extract cardiovascular-relevant features from transcriptomic data
3. **Model Development**: Build ML models using mission duration, age, and physiological markers
4. **Validation**: Use bed rest data (GSE14798/OSD-51) to validate terrestrial applications

## Directory Structure

```
microgravity_cardiovascular_data/
├── raw_data/
│   ├── astronaut_biomarkers/     # Direct human spaceflight data
│   ├── cellular_mechanisms/      # Mechanistic studies
│   └── bed_rest_analog/          # Earth-based validation data
├── processed_data/               # Cleaned and processed datasets
├── metadata/                     # Dataset documentation
├── analysis_scripts/             # Data analysis code
├── models/                       # ML models and results
└── results/                      # Final outputs and figures
```

## Dataset Priorities

**Priority 1 (Start Here):**
- OSD-575: Inspiration4 blood biomarkers
- GSE14798/OSD-51: Bed rest validation data

**Priority 2 (Mechanistic Understanding):**
- OSD-635: Vascular smooth muscle changes
- OSD-484: Plasma exosome biomarkers
- GSE137081/OSD-258: Cardiomyocyte responses
