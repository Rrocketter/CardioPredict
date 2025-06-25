#!/usr/bin/env python3
"""
Microgravity-Induced Cardiovascular Risk Prediction Model
Data Download and Organization Script

This script downloads and organizes datasets for predicting cardiovascular
risk in astronauts and bedridden patients based on physiological markers.
"""

import os
import requests
import pandas as pd
from pathlib import Path
import zipfile
import gzip
import shutil
import time
from urllib.parse import urljoin
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MicrogravityDataDownloader:
    """Downloads and organizes microgravity cardiovascular datasets"""
    
    def __init__(self, base_dir: str = "microgravity_cardiovascular_data"):
        self.base_dir = Path(base_dir)
        self.setup_directory_structure()
        
        # Dataset configurations
        self.datasets = {
            "OSD-575": {
                "name": "SpaceX_Inspiration4_Blood_Serum",
                "description": "SpaceX Inspiration4 Blood Serum Metabolic Panel and Immune/Cardiac Cytokine Arrays",
                "url_base": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-575",
                "files": [
                    "OSD-575_metadata.xlsx",
                    "OSD-575_assay_data.csv",
                    "OSD-575_sample_data.csv"
                ],
                "priority": 1,
                "data_type": "human_astronaut_biomarkers"
            },
            "OSD-635": {
                "name": "Vascular_Smooth_Muscle_Spaceflight",
                "description": "Spaceflight effects on human vascular smooth muscle cell phenotype and function",
                "url_base": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-635",
                "files": [
                    "OSD-635_metadata.xlsx",
                    "OSD-635_transcriptomics_data.csv",
                    "OSD-635_proteomics_data.csv"
                ],
                "priority": 2,
                "data_type": "human_vascular_cells"
            },
            "OSD-484": {
                "name": "Astronaut_Plasma_Exosomes",
                "description": "Astronauts plasma-derived exosomes induced gene expression in AC16 cells",
                "url_base": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-484",
                "files": [
                    "OSD-484_metadata.xlsx",
                    "OSD-484_gene_expression.csv",
                    "OSD-484_exosome_analysis.csv"
                ],
                "priority": 2,
                "data_type": "astronaut_plasma_biomarkers"
            },
            "GSE137081": {
                "name": "Cardiomyocyte_Spaceflight_iPSC",
                "description": "Effects of Spaceflight on Human iPSC-Derived Cardiomyocytes",
                "url_base": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137081",
                "geo_id": "GSE137081",
                "priority": 2,
                "data_type": "human_cardiomyocytes"
            },
            "OSD-258": {
                "name": "Cardiomyocyte_Spaceflight_OSD",
                "description": "Effects of Spaceflight on Human iPSC-Derived Cardiomyocytes (OSD version)",
                "url_base": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-258",
                "files": [
                    "OSD-258_metadata.xlsx",
                    "OSD-258_gene_expression.csv",
                    "OSD-258_cardiac_function.csv"
                ],
                "priority": 2,
                "data_type": "human_cardiomyocytes"
            },
            "GSE14798": {
                "name": "Bed_Rest_Skeletal_Muscle",
                "description": "Woman skeletal muscle transcriptome with bed rest and countermeasures",
                "url_base": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14798",
                "geo_id": "GSE14798",
                "priority": 1,
                "data_type": "human_bed_rest"
            },
            "OSD-51": {
                "name": "Bed_Rest_Skeletal_Muscle_OSD",
                "description": "Woman skeletal muscle transcriptome with bed rest (OSD version)",
                "url_base": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-51",
                "files": [
                    "OSD-51_metadata.xlsx",
                    "OSD-51_transcriptomics.csv",
                    "OSD-51_clinical_data.csv"
                ],
                "priority": 1,
                "data_type": "human_bed_rest"
            }
        }
    
    def setup_directory_structure(self):
        """Create organized directory structure"""
        dirs = [
            self.base_dir / "raw_data" / "astronaut_biomarkers",
            self.base_dir / "raw_data" / "cellular_mechanisms",
            self.base_dir / "raw_data" / "bed_rest_analog",
            self.base_dir / "processed_data",
            self.base_dir / "metadata",
            self.base_dir / "documentation",
            self.base_dir / "analysis_scripts",
            self.base_dir / "models",
            self.base_dir / "results"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def download_file(self, url: str, filepath: Path, max_retries: int = 3) -> bool:
        """Download a file with retry logic"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded: {filepath}")
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait before retry
                
        logger.error(f"Failed to download {url} after {max_retries} attempts")
        return False
    
    def download_geo_dataset(self, geo_id: str, dataset_name: str) -> bool:
        """Download GEO dataset using GEOquery-style URLs"""
        try:
            # Create dataset directory
            dataset_dir = self.base_dir / "raw_data" / self.get_data_category(dataset_name) / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Common GEO file URLs
            geo_urls = {
                "series_matrix": f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:-3]}nnn/{geo_id}/matrix/{geo_id}_series_matrix.txt.gz",
                "soft": f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:-3]}nnn/{geo_id}/soft/{geo_id}_family.soft.gz",
                "miniml": f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:-3]}nnn/{geo_id}/miniml/{geo_id}_family.xml.tgz"
            }
            
            success = True
            for file_type, url in geo_urls.items():
                filename = f"{geo_id}_{file_type}.{url.split('.')[-1]}"
                filepath = dataset_dir / filename
                
                if not self.download_file(url, filepath):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error downloading GEO dataset {geo_id}: {e}")
            return False
    
    def download_osdr_dataset(self, dataset_id: str, dataset_info: Dict) -> bool:
        """Download OSDR dataset"""
        try:
            dataset_name = dataset_info["name"]
            dataset_dir = self.base_dir / "raw_data" / self.get_data_category(dataset_name) / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Create README for dataset
            readme_content = f"""# {dataset_info['description']}

Dataset ID: {dataset_id}
Data Type: {dataset_info['data_type']}
Priority: {dataset_info['priority']}

## Files in this dataset:
"""
            
            success = True
            if "files" in dataset_info:
                for filename in dataset_info["files"]:
                    file_url = f"{dataset_info['url_base']}/{filename}"
                    filepath = dataset_dir / filename
                    
                    if not self.download_file(file_url, filepath):
                        success = False
                    
                    readme_content += f"- {filename}\n"
            
            # Save README
            readme_path = dataset_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            return success
            
        except Exception as e:
            logger.error(f"Error downloading OSDR dataset {dataset_id}: {e}")
            return False
    
    def get_data_category(self, dataset_name: str) -> str:
        """Categorize dataset based on its type"""
        if "Inspiration4" in dataset_name or "Exosomes" in dataset_name:
            return "astronaut_biomarkers"
        elif "Bed_Rest" in dataset_name:
            return "bed_rest_analog"
        else:
            return "cellular_mechanisms"
    
    def download_all_datasets(self):
        """Download all configured datasets"""
        logger.info("Starting download of all microgravity cardiovascular datasets...")
        
        # Sort by priority
        sorted_datasets = sorted(self.datasets.items(), key=lambda x: x[1]["priority"])
        
        download_summary = {
            "successful": [],
            "failed": []
        }
        
        for dataset_id, dataset_info in sorted_datasets:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {dataset_id}: {dataset_info['description']}")
            logger.info(f"{'='*50}")
            
            try:
                if dataset_id.startswith("GSE"):
                    success = self.download_geo_dataset(dataset_id, dataset_info["name"])
                else:
                    success = self.download_osdr_dataset(dataset_id, dataset_info)
                
                if success:
                    download_summary["successful"].append(dataset_id)
                    logger.info(f"✓ Successfully processed {dataset_id}")
                else:
                    download_summary["failed"].append(dataset_id)
                    logger.error(f"✗ Failed to process {dataset_id}")
                
                # Brief pause between datasets
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Unexpected error processing {dataset_id}: {e}")
                download_summary["failed"].append(dataset_id)
        
        self.create_download_summary(download_summary)
        return download_summary
    
    def create_download_summary(self, summary: Dict):
        """Create a summary report of the download process"""
        summary_path = self.base_dir / "download_summary.md"
        
        content = f"""# Microgravity Cardiovascular Data Download Summary

## Project Overview
This collection supports the development of an ML model to predict cardiovascular risk in astronauts and bedridden patients based on physiological markers from microgravity exposure.

## Download Results

### Successfully Downloaded ({len(summary['successful'])} datasets):
"""
        
        for dataset_id in summary["successful"]:
            dataset_info = self.datasets[dataset_id]
            content += f"- **{dataset_id}**: {dataset_info['description']}\n"
        
        content += f"\n### Failed Downloads ({len(summary['failed'])} datasets):\n"
        
        for dataset_id in summary["failed"]:
            dataset_info = self.datasets[dataset_id]
            content += f"- **{dataset_id}**: {dataset_info['description']}\n"
        
        content += f"""
## Next Steps

1. **Data Exploration**: Start with OSD-575 (Inspiration4 biomarkers) as your anchor dataset
2. **Feature Engineering**: Extract cardiovascular-relevant features from transcriptomic data
3. **Model Development**: Build ML models using mission duration, age, and physiological markers
4. **Validation**: Use bed rest data (GSE14798/OSD-51) to validate terrestrial applications

## Directory Structure

```
{self.base_dir}/
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
"""
        
        with open(summary_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Download summary saved to: {summary_path}")

def main():
    """Main execution function"""
    downloader = MicrogravityDataDownloader()
    
    print("🚀 Microgravity Cardiovascular Risk Prediction Data Downloader")
    print("=" * 60)
    print("This script will download datasets for predicting cardiovascular")
    print("risk in astronauts and bedridden patients.\n")
    
    # Ask user for confirmation
    response = input("Do you want to proceed with downloading all datasets? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Start download process
    summary = downloader.download_all_datasets()
    
    print("\n" + "=" * 60)
    print("🏁 Download Process Complete!")
    print(f"✓ Successful: {len(summary['successful'])} datasets")
    print(f"✗ Failed: {len(summary['failed'])} datasets")
    print(f"\nData saved to: {downloader.base_dir}")
    print("Check download_summary.md for detailed results.")

if __name__ == "__main__":
    main()