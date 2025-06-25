import requests
import json
import csv
import pandas as pd
from datetime import datetime
import time

class NASAOSDRSearcher:
    def __init__(self):
        # Correct API endpoint for GET requests
        self.search_endpoint = "https://osdr.nasa.gov/osdr/data/search"

    def search_datasets(self, search_terms, output_format='csv'):
        """
        Search NASA OSDR for datasets related to cardiovascular/astronaut health
        """
        all_results = []
        print("Starting NASA OSDR dataset search...")
        print(f"Search terms: {search_terms}")
        print("-" * 50)

        for term in search_terms:
            print(f"Searching for: '{term}'")
            try:
                # Parameters for the GET request.
                params = {
                    'term': term,
                    'size': 100 # Get up to 100 results per term
                }
                response = requests.get(self.search_endpoint, params=params, timeout=45)

                if response.status_code == 200:
                    data = response.json()
                    
                    # <<< KEY CHANGE HERE >>>
                    # Based on the inspection, the results are in data['hits']['hits']
                    if 'hits' in data and 'hits' in data['hits'] and data['hits']['hits']:
                        search_hits = data['hits']['hits']
                        print(f"Found {len(search_hits)} results for '{term}'")

                        for hit in search_hits:
                            # Pass the entire 'hit' object to the extractor
                            study_info = self.extract_study_info(hit, term)
                            all_results.append(study_info)
                    else:
                        print(f"No results found for '{term}'.")

                else:
                    print(f"Error searching for '{term}': HTTP {response.status_code}")
                    print(f"Response Body: {response.text}")

            except requests.exceptions.RequestException as e:
                print(f"A network error occurred while searching for '{term}': {str(e)}")
            except Exception as e:
                print(f"An unexpected error occurred while processing '{term}': {str(e)}")

            time.sleep(1) # Be respectful to NASA's servers

        # Remove duplicates based on study ID
        unique_results = []
        seen_ids = set()
        for result in all_results:
            if result['study_id'] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result['study_id'])

        print(f"\nTotal unique datasets found: {len(unique_results)}")

        # Export results
        if unique_results:
            self.export_results(unique_results, output_format)
            return unique_results
        else:
            print("No datasets found across all search terms.")
            return []

    def extract_study_info(self, hit, search_term):
        """Extract relevant information from a single search hit object."""
        # The actual data for each study is nested in the '_source' key
        source_data = hit.get('_source', {})

        study_id = source_data.get('Accession', 'N/A')
        title = source_data.get('Study Title', 'N/A')
        description = source_data.get('Study Description', 'N/A')
        
        # Organism can be a list, so we join it into a string
        organism_list = source_data.get('organism', [])
        organism_str = ", ".join(filter(None, organism_list)) # Joins non-empty strings

        return {
            'search_term_used': search_term,
            'study_id': study_id,
            'title': title,
            'description': description[:500] + '...' if description and len(description) > 500 else description,
            'organism': organism_str or 'N/A',
            'relevance_score': self.calculate_relevance_score(source_data),
            'url': f"https://osdr.nasa.gov/bio/repo/data/studies/{study_id}"
        }

    def calculate_relevance_score(self, source_data):
        """Calculate relevance score based on the _source data."""
        score = 0
        title = (source_data.get('Study Title') or '').lower()
        description = (source_data.get('Study Description') or '').lower()
        
        text_to_search = f"{title} {description}"

        high_keywords = ['cardiovascular', 'heart', 'cardiac', 'blood pressure', 'circulation', 'astronaut', 'spaceflight', 'microgravity', 'arterial']
        medium_keywords = ['physiological', 'human', 'health', 'medical', 'clinical', 'biomedical', 'inspiration4', 'artery']
        low_keywords = ['space', 'flight', 'mission', 'crew', 'iss', 'vascular']

        for keyword in high_keywords:
            if keyword in text_to_search: score += 10
        for keyword in medium_keywords:
            if keyword in text_to_search: score += 3
        for keyword in low_keywords:
            if keyword in text_to_search: score += 1

        # Give a large bonus for human studies, checking against the organism list
        organism_list = source_data.get('organism', [])
        if any('homo sapiens' in str(org).lower() for org in organism_list):
            score += 15

        return round(score, 1)

    def export_results(self, results, format_type):
        """Export results to the specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_sorted = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        filename_base = f"nasa_osdr_search_results_{timestamp}"

        if format_type == 'csv':
            filename = f"{filename_base}.csv"
            df = pd.DataFrame(results_sorted)
            df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
            print(f"\nResults successfully exported to {filename}")

        elif format_type == 'json':
            filename = f"{filename_base}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_sorted, f, indent=2, ensure_ascii=False)
            print(f"\nResults successfully exported to {filename}")

        elif format_type == 'txt':
            filename = f"{filename_base}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                # ... (code for writing to txt file)
                for i, result in enumerate(results_sorted, 1):
                    f.write(f"----- DATASET #{i} -----\n")
                    f.write(f"Relevance Score: {result['relevance_score']}\n")
                    f.write(f"Study ID: {result['study_id']}\n")
                    f.write(f"Title: {result['title']}\n")
                    f.write(f"Organism: {result['organism']}\n")
                    f.write(f"URL: {result['url']}\n")
                    f.write(f"Description: {result['description']}\n\n")
            print(f"\nResults successfully exported to {filename}")

def main():
    """Main function to run the search tool."""
    searcher = NASAOSDRSearcher()
    cardiovascular_terms = [
        "cardiovascular", "heart", "cardiac", "blood pressure", "circulation",
        "astronaut health", "human spaceflight", "microgravity human", "Inspiration4",
        "space medicine", "orthostatic intolerance", "arterial stiffness", "vascular"
    ]

    print("NASA OSDR Cardiovascular Dataset Search Tool")
    print("=" * 50)
    
    while True:
        format_choice = input("Choose output format (csv, json, or txt): ").lower().strip()
        if format_choice in ['csv', 'json', 'txt']: break
        print("Invalid choice. Please enter 'csv', 'json', or 'txt'.")

    results = searcher.search_datasets(cardiovascular_terms, format_choice)

    if results:
        print(f"\nSearch complete!")
        print("\n--- Top 5 Most Relevant Datasets ---")
        # The list is already sorted from the export function
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. (Score: {result['relevance_score']}) {result['title']}")
            print(f"   ID: {result['study_id']}, Organism: {result['organism']}")

        print(f"\nFull results have been saved. Please copy the text from the generated file")
        print("and paste it in your reply so I can help you choose the best datasets.")
    else:
        print("\nSearch complete, but no relevant datasets were found.")

if __name__ == "__main__":
    main()