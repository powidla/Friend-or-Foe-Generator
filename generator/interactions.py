import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csc_matrix, csr_matrix, hstack
from scipy.optimize import linprog
import json
import os
from datetime import datetime
import random
from tqdm import tqdm
from collections import defaultdict

from loading import load_model_simple
from modeling import *


def classify_interaction_detailed(m1_change, m2_change, tolerance=1e-16):
    if m1_change > tolerance and m2_change > tolerance:
        interaction_type = "Cooperative"
        category = "Mutualism"
    elif m1_change < -tolerance and m2_change < -tolerance:
        interaction_type = "Competitive"
        category = "Competition"
    elif abs(m1_change) < tolerance and abs(m2_change) < tolerance:
        interaction_type = "Obligate XX"
        category = "Obligate"
    elif m1_change > tolerance and m2_change < -tolerance:
        interaction_type = "Obligate PlusX"
        category = "Obligate"
    elif m1_change < -tolerance and m2_change > tolerance:
        interaction_type = "Obligate XPlus"
        category = "Obligate"
    else:
        interaction_type = "Neutral"
        category = "Neutral"
    return interaction_type, category


class TargetedInteractionSearcher:
    def __init__(self, microbe1, microbe2, target_interactions=None, max_environments_per_type=1000, interaction_tolerance=1e-16):
        self.microbe1 = microbe1
        self.microbe2 = microbe2
        self.target_interactions = target_interactions or [
            "Cooperative", "Competitive", "Obligate XX", 
            "Obligate PlusX", "Obligate XPlus", "Neutral"
        ]
        self.max_per_type = max_environments_per_type
        self.tolerance = interaction_tolerance
        self.found_interactions = defaultdict(list)
        self.interaction_counts = defaultdict(int)

        self.collector = FriendOrFoeDataCollector(microbe1, microbe2)
        print(f"Targeting interactions: {self.target_interactions}")
        print(f"Max environments per type: {self.max_per_type}")
    
    def is_target_complete(self):
        for interaction_type in self.target_interactions:
            if len(self.found_interactions[interaction_type]) < self.max_per_type:
                return False
        return True
    
    def get_search_progress(self):
        '''
        Get current search progress
        '''
        progress = {}
        for interaction_type in self.target_interactions:
            found = len(self.found_interactions[interaction_type])
            target = self.max_per_type
            progress[interaction_type] = f"{found}/{target}"
        return progress
    
    def search_for_target_interactions(self, max_attempts=50000, min_nutrients=200, max_nutrients=424, optimization_method='FBA', min_growth_rate=1e-16):
        num_compounds = self.microbe1["S_ext"].shape[0]
        pair_model = create_pair_model_simple(self.microbe1, self.microbe2)
        
        attempts = 0
        successful_tests = 0
        with tqdm(total=max_attempts, desc="Searching environments") as pbar:
            while attempts < max_attempts and not self.is_target_complete():
                attempts += 1
                env_id = f"search_{attempts:06d}"
                env_rhslb, available_nutrients = generate_random_environment(num_compounds, min_nutrients, max_nutrients)
                success, results = test_pair_growth_in_environment_flexible(
                    self.microbe1, self.microbe2, env_rhslb, pair_model,
                    optimization_method=optimization_method,
                    min_growth_rate=min_growth_rate
                )
                if success:
                    successful_tests += 1
                    m1_change = results['changes']['m1_change']
                    m2_change = results['changes']['m2_change']
                    
                    interaction_type, category = classify_interaction_detailed(m1_change, m2_change, self.tolerance)
                    if (interaction_type in self.target_interactions and 
                        len(self.found_interactions[interaction_type]) < self.max_per_type):
                        environment_data = {
                            'env_id': env_id,
                            'env_rhslb': env_rhslb,
                            'available_nutrients': available_nutrients,
                            'results': results,
                            'interaction_type': interaction_type,
                            'interaction_category': category,
                            'm1_change': m1_change,
                            'm2_change': m2_change,
                            'n_nutrients': len(available_nutrients)
                        }
                        self.found_interactions[interaction_type].append(environment_data)
                        self.collector.add_environment_result(env_id, env_rhslb, available_nutrients, results, pair_model)
                    self.interaction_counts[interaction_type] += 1

                if attempts % 1000 == 0:
                    progress = self.get_search_progress()
                    success_rate = successful_tests / attempts * 100
                    progress_str = " | ".join([f"{k}: {v}" for k, v in progress.items()])
                    pbar.set_postfix_str(f"Success: {success_rate:.1f}% | {progress_str}")
                
                pbar.update(1)
        
        print(f"\nSEARCH DONE!")
        print(f"Attempts: {attempts:,}")
        print(f"Successful tests: {successful_tests:,} ({successful_tests/attempts*100:.1f}%)")
        print(f"\nFOUND TARGET INTERACTIONS:")
        total_found = 0
        for interaction_type in self.target_interactions:
            found = len(self.found_interactions[interaction_type])
            total_found += found
            print(f"   {interaction_type}: {found}/{self.max_per_type}")
        
        print(f"\nALL INTERACTION STATISTICS:")
        for interaction_type, count in sorted(self.interaction_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / successful_tests * 100 if successful_tests > 0 else 0
            print(f"{interaction_type}: {count} ({percentage:.1f}%)")
        
        return {
            'attempts': attempts,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / attempts,
            'found_interactions': dict(self.found_interactions),
            'interaction_counts': dict(self.interaction_counts),
            'total_environments_saved': total_found
        }
    
    def create_interaction_folders(self, base_output_dir):
        interaction_folders = {}
        os.makedirs(base_output_dir, exist_ok=True)
        for interaction_type in self.found_interactions.keys():
            if len(self.found_interactions[interaction_type]) > 0:
                folder_name = interaction_type.lower().replace(' ', '_').replace('+', 'plus').replace('x', 'x')
                folder_path = os.path.join(base_output_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                interaction_folders[interaction_type] = folder_path
                
        return interaction_folders
    
    def save_targeted_results(self, output_dir="./targeted_interactions_output"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model1_clean = self.microbe1['name'].replace('.xml', '').replace(' ', '_')
        model2_clean = self.microbe2['name'].replace('.xml', '').replace(' ', '_')
        interaction_folders = self.create_interaction_folders(output_dir)
        
        saved_files = {}
        interaction_summaries = {}

        if len(self.collector.summary_data) > 0:
            files = self.collector.save_consolidated_csvs(output_dir, timestamp, model1_clean, model2_clean)
            saved_files.update(files)
        for interaction_type, environments in self.found_interactions.items():
            if len(environments) > 0:
                interaction_folder = interaction_folders[interaction_type]
                
                print(f"Saving {interaction_type} data to: {os.path.basename(interaction_folder)}/")
                summary_data = []
                environment_details = []
                for env_data in environments:
                    summary_row = {
                        'env_id': env_data['env_id'],
                        'interaction_type': env_data['interaction_type'],
                        'interaction_category': env_data['interaction_category'],
                        'n_nutrients': env_data['n_nutrients'],
                        'm1_solo_growth': env_data['results']['growth_rates']['m1_alone'],
                        'm2_solo_growth': env_data['results']['growth_rates']['m2_alone'],
                        'm1_pair_growth': env_data['results']['growth_rates']['m1_with_m2_nw'],
                        'm2_pair_growth': env_data['results']['growth_rates']['m2_with_m1_nw'],
                        'm1_change': env_data['m1_change'],
                        'm2_change': env_data['m2_change'],
                        'm1_change_percent': (env_data['m1_change'] / env_data['results']['growth_rates']['m1_alone'] * 100) 
                                           if env_data['results']['growth_rates']['m1_alone'] > 0 else 0,
                        'm2_change_percent': (env_data['m2_change'] / env_data['results']['growth_rates']['m2_alone'] * 100) 
                                           if env_data['results']['growth_rates']['m2_alone'] > 0 else 0
                    }
                    summary_data.append(summary_row)
                    
                    # Environment details
                    env_detail = {
                        'env_id': env_data['env_id'],
                        'n_nutrients': env_data['n_nutrients'],
                        'available_nutrients': env_data['available_nutrients']
                    }
                    environment_details.append(env_detail)
                summary_df = pd.DataFrame(summary_data)
                summary_filename = f"summary_{interaction_type}_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
                summary_path = os.path.join(interaction_folder, summary_filename)
                summary_df.to_csv(summary_path, index=False)
                saved_files[f'{interaction_type}_summary'] = os.path.join(os.path.basename(interaction_folder), summary_filename)
                
                env_details_df = pd.DataFrame(environment_details)
                env_filename = f"environments_{interaction_type}_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
                env_path = os.path.join(interaction_folder, env_filename)
                env_details_df.to_csv(env_path, index=False)
                saved_files[f'{interaction_type}_environments'] = os.path.join(os.path.basename(interaction_folder), env_filename)
                
                interaction_metadata = {
                    'interaction_type': interaction_type,
                    'count': len(environments),
                    'timestamp': timestamp,
                    'microbe1_name': self.microbe1['name'],
                    'microbe2_name': self.microbe2['name'],
                    'avg_m1_change': np.mean([env['m1_change'] for env in environments]),
                    'avg_m2_change': np.mean([env['m2_change'] for env in environments]),
                    'avg_nutrients': np.mean([env['n_nutrients'] for env in environments]),
                    'min_nutrients': min([env['n_nutrients'] for env in environments]),
                    'max_nutrients': max([env['n_nutrients'] for env in environments]),
                    'tolerance_used': self.tolerance
                }
                
                metadata_filename = f"metadata_{interaction_type}_{timestamp}.json"
                metadata_path = os.path.join(interaction_folder, metadata_filename)
            
                def convert_numpy(obj):
                    if isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj
                
                with open(metadata_path, 'w') as f:
                    json.dump(convert_numpy(interaction_metadata), f, indent=2)
                saved_files[f'{interaction_type}_metadata'] = os.path.join(os.path.basename(interaction_folder), metadata_filename)
                
                # Store summary for master file
                interaction_summaries[interaction_type] = {
                    'count': len(environments),
                    'avg_m1_change': np.mean([env['m1_change'] for env in environments]),
                    'avg_m2_change': np.mean([env['m2_change'] for env in environments]),
                    'avg_nutrients': np.mean([env['n_nutrients'] for env in environments]),
                    'min_nutrients': min([env['n_nutrients'] for env in environments]),
                    'max_nutrients': max([env['n_nutrients'] for env in environments]),
                    'folder': os.path.basename(interaction_folder)
                }
        
        master_summary = {
            'timestamp': timestamp,
            'microbe1_name': self.microbe1['name'],
            'microbe2_name': self.microbe2['name'],
            'target_interactions': self.target_interactions,
            'max_environments_per_type': self.max_per_type,
            'interaction_tolerance': self.tolerance,
            'found_interactions_summary': interaction_summaries,
            'total_environments_saved': sum(len(envs) for envs in self.found_interactions.values()),
            'saved_files': saved_files,
            'folder_structure': {interaction_type: os.path.basename(folder) for interaction_type, folder in interaction_folders.items()}
        }

def search_specific_interactions(model1_path, model2_path, target_interactions=["Cooperative", "Competitive"], max_per_type=500, max_attempts=20000, optimization_method='FBA', output_dir="./targeted_interactions"):
    microbe1 = load_model_simple(model1_path)
    microbe2 = load_model_simple(model2_path)
    searcher = TargetedInteractionSearcher(microbe1, microbe2, target_interactions=target_interactions, max_environments_per_type=max_per_type)
    search_results = searcher.search_for_target_interactions(max_attempts=max_attempts, optimization_method=optimization_method)
    saved_files, summary = searcher.save_targeted_results(output_dir)
    return search_results, saved_files, summary
  
