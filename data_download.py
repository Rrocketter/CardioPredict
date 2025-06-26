#!/usr/bin/env python3
"""
NASA OSDR Dataset Downloader for Microgravity Cardiovascular Risk Prediction Project
Downloads specified OSD datasets and logs all files for proper file management.
"""

import requests
import os
import time
from datetime import datetime
from urllib.parse import urlparse
import json

class OSDRDownloader:
    def __init__(self, base_dir="cardiovascular_risk_data"):
        self.base_url = "https://osdr.nasa.gov"
        self.base_dir = base_dir
        self.log_file = "download_log.txt"
        self.datasets = {
            "OSD-575": "SpaceX Inspiration4 Blood Serum Metabolic Panel and Immune/Cardiac Cytokine Arrays",
            "OSD-635": "Spaceflight effects on human vascular smooth muscle cell phenotype and function", 
            "OSD-484": "Astronauts plasma-derived exosomes induced gene expression in AC16 cells",
            "OSD-258": "Effects of Spaceflight on Human Induced Pluripotent Stem Cell-Derived Cardiomyocyte",
            "OSD-51": "Woman skeletal muscle transcriptome with bed rest and countermeasures"
        }
        
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"NASA OSDR Download Log - Started: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
    
    def log_message(self, message):
        """Log message to both console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: {message}\n")
    
    def get_study_files(self, osd_id):
        """Get file list for a specific OSD study"""
        url = f"{self.base_url}/osdr/data/osd/files/{osd_id}"
        
        try:
            self.log_message(f"Fetching file list for {osd_id}...")
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.log_message(f"Error fetching file list for {osd_id}: {e}")
            return None
    
    def get_study_metadata(self, osd_id):
        """Get metadata for a specific OSD study"""
        url = f"{self.base_url}/osdr/data/osd/meta/{osd_id}"
        
        try:
            self.log_message(f"Fetching metadata for {osd_id}...")
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.log_message(f"Error fetching metadata for {osd_id}: {e}")
            return None
    
    def download_file(self, file_info, study_dir, osd_id):
        """Download individual file"""
        file_name = file_info['file_name']
        remote_url = file_info['remote_url']
        file_size = file_info.get('file_size', 0)
        category = file_info.get('category', 'Unknown')
        
        # Create full download URL
        download_url = f"{self.base_url}{remote_url}"
        
        # Create category subdirectory
        category_dir = os.path.join(study_dir, category.replace(" ", "_").replace("/", "_"))
        os.makedirs(category_dir, exist_ok=True)
        
        file_path = os.path.join(category_dir, file_name)
        
        # Skip if file already exists
        if os.path.exists(file_path):
            self.log_message(f"  SKIP: {file_name} already exists")
            return True
        
        try:
            self.log_message(f"  Downloading: {file_name} ({file_size:,} bytes)")
            
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify file size
            actual_size = os.path.getsize(file_path)
            if file_size > 0 and actual_size != file_size:
                self.log_message(f"  WARNING: Size mismatch for {file_name}. Expected: {file_size}, Got: {actual_size}")
            
            self.log_message(f"  SUCCESS: Downloaded {file_name}")
            return True
            
        except requests.exceptions.RequestException as e:
            self.log_message(f"  ERROR: Failed to download {file_name}: {e}")
            return False
        except Exception as e:
            self.log_message(f"  ERROR: Unexpected error downloading {file_name}: {e}")
            return False
    
    def download_study(self, osd_id):
        """Download all files for a specific study"""
        self.log_message(f"\n{'='*60}")
        self.log_message(f"Processing {osd_id}: {self.datasets.get(osd_id, 'Unknown Study')}")
        self.log_message(f"{'='*60}")
        
        # Create study directory
        study_dir = os.path.join(self.base_dir, osd_id)
        os.makedirs(study_dir, exist_ok=True)
        
        # Get and save metadata
        metadata = self.get_study_metadata(osd_id)
        if metadata:
            metadata_file = os.path.join(study_dir, f"{osd_id}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.log_message(f"Saved metadata to: {metadata_file}")
        
        # Get file list
        file_data = self.get_study_files(osd_id)
        if not file_data or not file_data.get('success'):
            self.log_message(f"Failed to get file list for {osd_id}")
            return False
        
        studies = file_data.get('studies', {})
        if osd_id not in studies:
            self.log_message(f"No study data found for {osd_id}")
            return False
        
        study_info = studies[osd_id]
        file_count = study_info.get('file_count', 0)
        study_files = study_info.get('study_files', [])
        
        self.log_message(f"Found {file_count} files for {osd_id}")
        
        # Track download statistics
        success_count = 0
        total_files = len(study_files)
        
        # Download each file
        for i, file_info in enumerate(study_files, 1):
            self.log_message(f"[{i}/{total_files}] Processing file...")
            if self.download_file(file_info, study_dir, osd_id):
                success_count += 1
            
            # Add small delay to be respectful to the server
            time.sleep(0.5)
        
        # Log summary for this study
        self.log_message(f"\nStudy {osd_id} Summary:")
        self.log_message(f"  Total files: {total_files}")
        self.log_message(f"  Successfully downloaded: {success_count}")
        self.log_message(f"  Failed: {total_files - success_count}")
        
        return success_count > 0
    
    def create_study_summary(self):
        """Create a summary of all downloaded studies"""
        summary_file = os.path.join(self.base_dir, "study_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("Microgravity Cardiovascular Risk Prediction Project - Dataset Summary\n")
            f.write("="*70 + "\n\n")
            
            for osd_id, description in self.datasets.items():
                f.write(f"Dataset: {osd_id}\n")
                f.write(f"Description: {description}\n")
                
                study_dir = os.path.join(self.base_dir, osd_id)
                if os.path.exists(study_dir):
                    # Count files in each category
                    categories = {}
                    for root, dirs, files in os.walk(study_dir):
                        if root != study_dir:  # Skip the root directory
                            category = os.path.basename(root)
                            categories[category] = len([f for f in files if not f.endswith('.json')])
                    
                    f.write(f"Status: Downloaded\n")
                    f.write(f"File Categories:\n")
                    for category, count in categories.items():
                        f.write(f"  - {category}: {count} files\n")
                else:
                    f.write(f"Status: Not downloaded\n")
                
                f.write("\n" + "-"*50 + "\n\n")
        
        self.log_message(f"Study summary created: {summary_file}")
    
    def download_all(self):
        """Download all datasets"""
        self.log_message("Starting download of all cardiovascular risk datasets...")
        self.log_message(f"Download directory: {os.path.abspath(self.base_dir)}")
        
        successful_downloads = 0
        
        for osd_id in self.datasets.keys():
            try:
                if self.download_study(osd_id):
                    successful_downloads += 1
            except Exception as e:
                self.log_message(f"Unexpected error processing {osd_id}: {e}")
        
        # Create summary
        self.create_study_summary()
        
        # Final log
        self.log_message(f"\n{'='*60}")
        self.log_message(f"DOWNLOAD COMPLETE")
        self.log_message(f"{'='*60}")
        self.log_message(f"Total studies processed: {len(self.datasets)}")
        self.log_message(f"Successful downloads: {successful_downloads}")
        self.log_message(f"Failed downloads: {len(self.datasets) - successful_downloads}")
        self.log_message(f"Data directory: {os.path.abspath(self.base_dir)}")
        self.log_message(f"Log file: {os.path.abspath(self.log_file)}")


def main():
    """Main function to run the downloader"""
    print("NASA OSDR Dataset Downloader")
    print("Microgravity-Induced Cardiovascular Risk Prediction Project")
    print("="*60)
    
    # Initialize downloader
    downloader = OSDRDownloader()
    
    # Start download process
    try:
        downloader.download_all()
        print("\nDownload process completed successfully!")
        print(f"Check '{downloader.log_file}' for detailed logs.")
        print(f"Data saved to: {os.path.abspath(downloader.base_dir)}")
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()