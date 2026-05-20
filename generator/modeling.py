import numpy as np
import pandas as pd
from scipy.optimize import linprog
from datetime import datetime
import random
import os
from tqdm import tqdm
import cvxpy as cp


def generate_random_environment(num_compounds, min_nutrients=50, max_nutrients=424, concentration=-1000.0):
    '''
    Generate random nutrient environment
    '''
    n_nutrients = random.randint(min_nutrients, max_nutrients)
    available_nutrients = random.sample(range(num_compounds), n_nutrients)
    
    env_rhslb = np.zeros(num_compounds, dtype=np.float64)
    env_rhslb[available_nutrients] = concentration
    
    return env_rhslb, available_nutrients


def optimize_single_microbe_flexible(microbe, env_rhslb, method='FBA', regularization_weight=0.001):
    S_ext = microbe["S_ext"]
    S_int = microbe["S_int"]
    lb = microbe["lb"].flatten().astype(np.float64)
    ub = microbe["ub"].flatten().astype(np.float64)
    rhs_ext_ub = microbe["rhs_ext_ub"].flatten().astype(np.float64)
    rhs_int_lb = microbe["rhs_int_lb"].flatten().astype(np.float64)
    bmi = int(microbe["bmi"].flatten()[0]) - 1
    
    n_vars = len(lb)
    S_ext_dense = S_ext.toarray()
    S_int_dense = S_int.toarray()
    
    # Set up constraints 
    A_ub = np.vstack([-S_ext_dense, S_ext_dense])
    b_ub = np.concatenate([-env_rhslb, rhs_ext_ub])
    A_eq = S_int_dense
    b_eq = rhs_int_lb
    bounds = [(lb[i], ub[i]) for i in range(n_vars)]
    
    if method == 'FBA':
        c = np.zeros(n_vars)
        c[bmi] = -1.0
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'disp': False})
        
        if result.success:
            return {
                'growth_rate': float(abs(result.fun)),
                'fluxes': result.x,
                'env_fluxes': S_ext_dense @ result.x,
                'status': 'success',
                'optimization_info': {'method': 'FBA'}
            }
        else:
            return {'growth_rate': 0.0, 'fluxes': np.zeros(n_vars), 
                   'env_fluxes': np.zeros(len(env_rhslb)), 'status': 'failed'}
    
    elif method in ['MOMA', 'pFBA', 'L1_reg', 'L2_reg']:
        try:
            # Use CVXPY
            v = cp.Variable(n_vars)
            # Basic constraints
            constraints = [
                A_ub @ v <= b_ub,
                v >= np.array([b[0] for b in bounds]),
                v <= np.array([b[1] for b in bounds])
            ]
            if A_eq.size > 0:
                constraints.append(A_eq @ v == b_eq)
            
            # Method-specific objectives
            if method == 'MOMA':
                # Get reference using FBA first
                ref_result = optimize_single_microbe_flexible(microbe, env_rhslb, method='FBA')
                if ref_result['status'] == 'success':
                    ref_fluxes = ref_result['fluxes']
                    constraints.append(v[bmi] >= 0.9 * ref_fluxes[bmi])
                    objective = cp.Minimize(cp.sum_squares(v - ref_fluxes))
                else:
                    objective = cp.Maximize(v[bmi])  # Fall back to FBA
            elif method == 'pFBA':
                # Two-step: maximize growth, minimize flux
                objective = cp.Maximize(v[bmi] - 0.001 * cp.sum(cp.abs(v)))
            elif method == 'L1_reg':
                objective = cp.Maximize(v[bmi] - regularization_weight * cp.sum(cp.abs(v)))
            elif method == 'L2_reg':
                objective = cp.Maximize(v[bmi] - regularization_weight * cp.sum_squares(v))
            
            # Solve
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if prob.status == cp.OPTIMAL:
                fluxes = v.value
                return {
                    'growth_rate': float(fluxes[bmi]),
                    'fluxes': fluxes,
                    'env_fluxes': S_ext_dense @ fluxes,
                    'status': 'success',
                    'optimization_info': {'method': method, 'solver_status': 'optimal'}
                }
            else:
                return optimize_single_microbe_flexible(microbe, env_rhslb, method='FBA')
        
        except Exception as e:
            return optimize_single_microbe_flexible(microbe, env_rhslb, method='FBA')
    
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def optimize_single_microbe_simple(microbe, env_rhslb):
    return optimize_single_microbe_flexible(microbe, env_rhslb, method='FBA')


def optimize_single_microbe_simple(microbe, env_rhslb):
    '''
    Optimize single microbe with given environment
    '''
    S_ext = microbe["S_ext"]
    S_int = microbe["S_int"]
    lb = microbe["lb"].flatten().astype(np.float64)
    ub = microbe["ub"].flatten().astype(np.float64)
    rhs_ext_ub = microbe["rhs_ext_ub"].flatten().astype(np.float64)
    rhs_int_lb = microbe["rhs_int_lb"].flatten().astype(np.float64)
    bmi = int(microbe["bmi"].flatten()[0]) - 1
    
    n_vars = len(lb)
    num_ec = len(env_rhslb)
    
    c = np.zeros(n_vars)
    c[bmi] = -1.0
    
    S_ext_dense = S_ext.toarray()
    S_int_dense = S_int.toarray()
    
    A_ub = np.vstack([-S_ext_dense, S_ext_dense])
    b_ub = np.concatenate([-env_rhslb, rhs_ext_ub])
    
    A_eq = S_int_dense
    b_eq = rhs_int_lb
    
    bounds = [(lb[i], ub[i]) for i in range(n_vars)]

    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'disp': False})
    
    if result.success:
        growth_rate = abs(result.fun)
        fluxes = result.x
        env_fluxes = S_ext_dense @ fluxes
        
        return {
            'growth_rate': float(growth_rate),
            'fluxes': fluxes,
            'env_fluxes': env_fluxes,
            'status': 'success'
        }
    else:
        return {
            'growth_rate': 0.0,
            'fluxes': np.zeros(n_vars),
            'env_fluxes': np.zeros(num_ec),
            'status': 'failed',
            'message': result.message
        }

def create_pair_model_simple(microbe1, microbe2):
    '''
    Create combined model for pair analysis
    '''
    nrM1 = len(microbe1["lb"])
    nrM2 = len(microbe2["lb"])
    num_ec = microbe1["S_ext"].shape[0]
    num_nec = microbe1["S_int"].shape[0]
    
    lb_combined = np.concatenate([microbe1["lb"].flatten(), microbe2["lb"].flatten()]).astype(np.float64)
    
    ub_combined = np.concatenate([microbe1["ub"].flatten(), microbe2["ub"].flatten()]).astype(np.float64)
    
    rhs_ext_ub_combined = microbe1["rhs_ext_ub"].flatten() + microbe2["rhs_ext_ub"].flatten()
    S_ext1 = microbe1["S_ext"].toarray()
    S_ext2 = microbe2["S_ext"].toarray()
    S_int1 = microbe1["S_int"].toarray()
    S_int2 = microbe2["S_int"].toarray()
    
    S_ext_combined = np.hstack([S_ext1, S_ext2])
    S_int1_combined = np.hstack([S_int1, np.zeros((num_nec, nrM2))])
    S_int2_combined = np.hstack([np.zeros((num_nec, nrM1)), S_int2])
    
    bmi1 = int(microbe1["bmi"].flatten()[0]) - 1
    bmi2 = nrM1 + int(microbe2["bmi"].flatten()[0]) - 1
    
    return {
        'S_ext_combined': S_ext_combined,
        'S_int1_combined': S_int1_combined,
        'S_int2_combined': S_int2_combined,
        'lb_combined': lb_combined,
        'ub_combined': ub_combined,
        'rhs_ext_ub_combined': rhs_ext_ub_combined,
        'rhs_int_lb': np.zeros(num_nec),
        'bmi1': bmi1,
        'bmi2': bmi2,
        'nrM1': nrM1,
        'nrM2': nrM2
    }


def optimize_pair_with_constraint_simple(pair_model, env_rhslb, target_microbe, constraint_microbe, constraint_value, nw_tol=0.001):
    '''
    Optimize pair with 'no worse' constraint
    '''
    n_vars = len(pair_model['lb_combined'])
    num_ec = len(env_rhslb)
    
    c = np.zeros(n_vars)
    if target_microbe == 1:
        c[pair_model['bmi1']] = -1.0
    else:
        c[pair_model['bmi2']] = -1.0
    
    S_ext = pair_model['S_ext_combined']
    A_ub_env = np.vstack([-S_ext, S_ext])
    b_ub_env = np.concatenate([-env_rhslb, pair_model['rhs_ext_ub_combined']])
    
    no_worse_constraint = np.zeros(n_vars)
    if constraint_microbe == 1:
        no_worse_constraint[pair_model['bmi1']] = -1.0
    else:
        no_worse_constraint[pair_model['bmi2']] = -1.0
    
    A_ub = np.vstack([A_ub_env, no_worse_constraint.reshape(1, -1)])
    b_ub = np.concatenate([b_ub_env, [-(constraint_value - nw_tol)]])
    
    A_eq = np.vstack([pair_model['S_int1_combined'], pair_model['S_int2_combined']])
    b_eq = np.concatenate([pair_model['rhs_int_lb'], pair_model['rhs_int_lb']])
    
    bounds = [(pair_model['lb_combined'][i], pair_model['ub_combined'][i]) for i in range(n_vars)]
    
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'disp': False})
    
    if result.success:
        target_rate = abs(result.fun)
        fluxes = result.x
        if constraint_microbe == 1:
            constraint_rate = fluxes[pair_model['bmi1']]
        else:
            constraint_rate = fluxes[pair_model['bmi2']]
        env_fluxes = S_ext @ fluxes
        
        return {
            'target_rate': float(target_rate),
            'constraint_rate': float(constraint_rate),
            'env_fluxes': env_fluxes,
            'fluxes': fluxes,
            'status': 'success'
        }
    else:
        return {
            'target_rate': 0.0,
            'constraint_rate': 0.0,
            'env_fluxes': np.zeros(num_ec),
            'fluxes': np.zeros(n_vars),
            'status': 'failed',
            'message': result.message
        }

      
def test_pair_growth_in_environment_flexible(microbe1, microbe2, env_rhslb, pair_model, optimization_method='FBA', regularization_weight=0.001, min_growth_rate=1e-6):
    # Test solo growth with specified method
    solo1 = optimize_single_microbe_flexible(microbe1, env_rhslb, method=optimization_method, regularization_weight=regularization_weight)
    solo2 = optimize_single_microbe_flexible(microbe2, env_rhslb, method=optimization_method, regularization_weight=regularization_weight)
    
    if (solo1['status'] != 'success' or solo1['growth_rate'] < min_growth_rate or
        solo2['status'] != 'success' or solo2['growth_rate'] < min_growth_rate):
        return False, None
    
    result_m1_opt = optimize_pair_with_constraint_simple(
        pair_model, env_rhslb, 1, 2, solo2['growth_rate']
    )
    
    result_m2_opt = optimize_pair_with_constraint_simple(
        pair_model, env_rhslb, 2, 1, solo1['growth_rate']
    )
    
    if (result_m1_opt['status'] != 'success' or result_m2_opt['status'] != 'success'):
        return False, None
    
    growth_rates = {
        'm1_alone': solo1['growth_rate'],
        'm1_with_m2_nw': result_m1_opt['target_rate'],
        'm1_when_m2_opt': result_m2_opt['constraint_rate'],
        'm2_alone': solo2['growth_rate'],
        'm2_when_m1_opt': result_m1_opt['constraint_rate'],
        'm2_with_m1_nw': result_m2_opt['target_rate']
    }
    
    m1_change = growth_rates['m1_with_m2_nw'] - growth_rates['m1_alone']
    m2_change = growth_rates['m2_with_m1_nw'] - growth_rates['m2_alone']
    
    if m1_change > 0.001 and m2_change > 0.001:
        interaction_type = "Cooperative"
    elif m1_change < -0.001 and m2_change < -0.001:
        interaction_type = "Competitive"
    elif abs(m1_change) < 0.001 and abs(m2_change) < 0.001:
        interaction_type = "Obligate XX"
    elif m1_change > 0.001 and m2_change < -0.001:
        interaction_type = "Obligate PlusX"
    elif m1_change < -0.001 and m2_change > 0.001:
        interaction_type = "Obligate XPlus"
    else:
        interaction_type = "Neutral"
    
    results = {
        'growth_rates': growth_rates,
        'changes': {'m1_change': float(m1_change), 'm2_change': float(m2_change)},
        'interaction_type': interaction_type,
        'optimization_method': optimization_method,  # NEW: Track method used
        'solo_results': {'microbe1': solo1, 'microbe2': solo2},
        'pair_results': {'m1_optimized': result_m1_opt, 'm2_optimized': result_m2_opt}
    }
    return True, results


def test_pair_growth_in_environment(microbe1, microbe2, env_rhslb, pair_model, min_growth_rate=1e-6):
    return test_pair_growth_in_environment_flexible(microbe1, microbe2, env_rhslb, pair_model, optimization_method='FBA', min_growth_rate=min_growth_rate)


def test_pair_growth_in_environment(microbe1, microbe2, env_rhslb, pair_model, min_growth_rate=1e-6):
    '''
    Test if both microbes can grow in the given environment
    '''
    # Test solo growth
    solo1 = optimize_single_microbe_simple(microbe1, env_rhslb)
    solo2 = optimize_single_microbe_simple(microbe2, env_rhslb)
    
    if (solo1['status'] != 'success' or solo1['growth_rate'] < min_growth_rate or
        solo2['status'] != 'success' or solo2['growth_rate'] < min_growth_rate):
        return False, None
    
    # Test pair interactions
    result_m1_opt = optimize_pair_with_constraint_simple(pair_model, env_rhslb, 1, 2, solo2['growth_rate'])
    result_m2_opt = optimize_pair_with_constraint_simple(pair_model, env_rhslb, 2, 1, solo1['growth_rate'])
    
    if (result_m1_opt['status'] != 'success' or result_m2_opt['status'] != 'success'):
        return False, None
    
    # Compile results
    growth_rates = {
        'm1_alone': solo1['growth_rate'],
        'm1_with_m2_nw': result_m1_opt['target_rate'],
        'm1_when_m2_opt': result_m2_opt['constraint_rate'],
        'm2_alone': solo2['growth_rate'],
        'm2_when_m1_opt': result_m1_opt['constraint_rate'],
        'm2_with_m1_nw': result_m2_opt['target_rate']
    }
    
    m1_change = growth_rates['m1_with_m2_nw'] - growth_rates['m1_alone']
    m2_change = growth_rates['m2_with_m1_nw'] - growth_rates['m2_alone']
    
    # Classify interaction types based on Friend or Foe approach
    if m1_change > 0.001 and m2_change > 0.001:
        interaction_type = "Cooperative"
    elif m1_change < -0.001 and m2_change < -0.001:
        interaction_type = "Competitive"
    elif abs(m1_change) < 0.001 and abs(m2_change) < 0.001:
        interaction_type = "Obligate XX"
    elif m1_change > 0.001 and m2_change < -0.001:
        interaction_type = "Obligate PlusX"
    elif m1_change < -0.001 and m2_change > 0.001:
        interaction_type = "Obligate XPlus"
    else:
        interaction_type = "Neutral"
    
    results = {
        'growth_rates': growth_rates,
        'changes': {'m1_change': float(m1_change), 'm2_change': float(m2_change)},
        'interaction_type': interaction_type,
        'solo_results': {'microbe1': solo1, 'microbe2': solo2},
        'pair_results': {'m1_optimized': result_m1_opt, 'm2_optimized': result_m2_opt}
    }
    
    return True, results


class FriendOrFoeDataCollector:
    '''
    Collects data for consolidated CSV output
    '''
    def __init__(self, microbe1, microbe2, consumption_threshold=1e-6):
        self.microbe1 = microbe1
        self.microbe2 = microbe2
        self.consumption_threshold = consumption_threshold  
        # Get Matrix shapes
        self.m = microbe1["S_ext"].shape[0]  # external compounds
        self.n = microbe1["S_ext"].shape[1]  # microbe1 reactions
        self.p = microbe1["S_int"].shape[0]  # microbe1 internal compounds
        self.l = microbe2["S_ext"].shape[1]  # microbe2 reactions  
        self.h = microbe2["S_int"].shape[0]  # microbe2 internal compounds
        self.reset_collectors()
    
    def reset_collectors(self):
        '''
        Reset all data collectors
        '''
        self.environments_data = []
        self.summary_data = []
        self.solo_m1_reactions_data = []
        self.solo_m1_ext_compounds_data = []  # Binary: which external compounds consumed
        self.solo_m1_int_compounds_data = []  # Binary: which internal compounds produced/consumed
        self.solo_m2_reactions_data = []
        self.solo_m2_ext_compounds_data = []  # Binary: which external compounds consumed
        self.solo_m2_int_compounds_data = []  # Binary: which internal compounds produced/consumed
        self.pair_m1opt_reactions_data = []
        self.pair_m1opt_compounds_data = []   # Binary: which compounds used in pair
        self.pair_m2opt_reactions_data = []
        self.pair_m2opt_compounds_data = []   # Binary: which compounds used in pair
    
    def flux_to_binary(self, flux_vector, mode='consumption'):
        
        if mode == 'consumption':
            return (flux_vector < -self.consumption_threshold).astype(int)
        elif mode == 'production':
            return (flux_vector > self.consumption_threshold).astype(int)
        elif mode == 'activity':
            return (np.abs(flux_vector) > self.consumption_threshold).astype(int)
        else:
            raise ValueError("Mode must be 'consumption', 'production', or 'activity'")
    
    def add_environment_result(self, env_id, env_rhslb, available_nutrients, results, pair_model):
        '''
        Add results from one environment to all collectors with binary compound data
        '''
        env_row = {'env_id': env_id, 'n_nutrients': len(available_nutrients)}
        for i in range(self.m):
            env_row[f'C_ext_{i+1}'] = env_rhslb[i]
        self.environments_data.append(env_row)
        # Summary 
        summary_row = {
            'env_id': env_id,
            'n_nutrients': len(available_nutrients),
            'interaction_type': results['interaction_type'],
            'm1_solo_growth': results['growth_rates']['m1_alone'],
            'm2_solo_growth': results['growth_rates']['m2_alone'],
            'm1_pair_growth': results['growth_rates']['m1_with_m2_nw'],
            'm2_pair_growth': results['growth_rates']['m2_with_m1_nw'],
            'm1_change': results['changes']['m1_change'],
            'm2_change': results['changes']['m2_change'],
            'm1_change_percent': (results['changes']['m1_change'] / results['growth_rates']['m1_alone'] * 100) if results['growth_rates']['m1_alone'] > 0 else 0,
            'm2_change_percent': (results['changes']['m2_change'] / results['growth_rates']['m2_alone'] * 100) if results['growth_rates']['m2_alone'] > 0 else 0
        }
        self.summary_data.append(summary_row)
        
        # Extract individual results
        solo1 = results['solo_results']['microbe1']
        solo2 = results['solo_results']['microbe2']
        pair_m1_opt = results['pair_results']['m1_optimized']
        pair_m2_opt = results['pair_results']['m2_optimized']
        
        # =================================
        # SOLO M1 DATA
        # =================================
        m1_reactions_row = {'env_id': env_id}
        for i in range(self.n):
            m1_reactions_row[f'R1_{i+1}'] = solo1['fluxes'][i]
        self.solo_m1_reactions_data.append(m1_reactions_row)

        S1_ext_dense = self.microbe1["S_ext"].toarray()
        m1_ext_compounds = S1_ext_dense @ solo1['fluxes']
        m1_ext_binary = self.flux_to_binary(m1_ext_compounds, mode='consumption')
        m1_ext_row = {'env_id': env_id}
        for i in range(self.m):
            m1_ext_row[f'C_ext_{i+1}'] = m1_ext_binary[i]
        self.solo_m1_ext_compounds_data.append(m1_ext_row)
        
        S1_int_dense = self.microbe1["S_int"].toarray()
        m1_int_compounds = S1_int_dense @ solo1['fluxes']
        m1_int_binary = self.flux_to_binary(m1_int_compounds, mode='activity')
        m1_int_row = {'env_id': env_id}
        for i in range(self.p):
            m1_int_row[f'C1_int_{i+1}'] = m1_int_binary[i]
        self.solo_m1_int_compounds_data.append(m1_int_row)
        
        # =================================
        # SOLO M2 DATA  
        # =================================
        m2_reactions_row = {'env_id': env_id}
        for i in range(self.l):
            m2_reactions_row[f'R2_{i+1}'] = solo2['fluxes'][i]
        self.solo_m2_reactions_data.append(m2_reactions_row)
    
        S2_ext_dense = self.microbe2["S_ext"].toarray()
        m2_ext_compounds = S2_ext_dense @ solo2['fluxes']
        m2_ext_binary = self.flux_to_binary(m2_ext_compounds, mode='consumption')
        m2_ext_row = {'env_id': env_id}
        for i in range(self.m):
            m2_ext_row[f'C_ext_{i+1}'] = m2_ext_binary[i]
        self.solo_m2_ext_compounds_data.append(m2_ext_row)
        
        S2_int_dense = self.microbe2["S_int"].toarray()
        m2_int_compounds = S2_int_dense @ solo2['fluxes']
        m2_int_binary = self.flux_to_binary(m2_int_compounds, mode='activity')
        m2_int_row = {'env_id': env_id}
        for i in range(self.h):
            m2_int_row[f'C2_int_{i+1}'] = m2_int_binary[i]
        self.solo_m2_int_compounds_data.append(m2_int_row)
        
        # =================================
        # PAIR M1 OPTIMIZED DATA
        # =================================
        pair_m1_reactions_row = {'env_id': env_id}
        for i in range(self.n + self.l):
            pair_m1_reactions_row[f'R_{i+1}'] = pair_m1_opt['fluxes'][i]
        self.pair_m1opt_reactions_data.append(pair_m1_reactions_row)
        
        S_combined = np.vstack([
            pair_model['S_ext_combined'],           # (m, n+l)
            pair_model['S_int1_combined'],          # (p, n+l) 
            pair_model['S_int2_combined']           # (h, n+l)
        ])
        pair_m1_compounds = S_combined @ pair_m1_opt['fluxes']
        pair_m1_ext_part = pair_m1_compounds[:self.m]  # External compounds
        pair_m1_int1_part = pair_m1_compounds[self.m:self.m+self.p]  # M1 internal
        pair_m1_int2_part = pair_m1_compounds[self.m+self.p:]  # M2 internal
        
        # Convert to binary
        pair_m1_ext_binary = self.flux_to_binary(pair_m1_ext_part, mode='consumption')
        pair_m1_int1_binary = self.flux_to_binary(pair_m1_int1_part, mode='activity')
        pair_m1_int2_binary = self.flux_to_binary(pair_m1_int2_part, mode='activity')
        pair_m1_compounds_binary = np.concatenate([pair_m1_ext_binary, pair_m1_int1_binary, pair_m1_int2_binary])
        pair_m1_compounds_row = {'env_id': env_id}
        for i in range(self.h + self.m + self.p):
            pair_m1_compounds_row[f'C_all_{i+1}'] = pair_m1_compounds_binary[i]
        self.pair_m1opt_compounds_data.append(pair_m1_compounds_row)
    
        # =================================
        # PAIR M2 OPTIMIZED DATA
        # =================================
        pair_m2_reactions_row = {'env_id': env_id}
        for i in range(self.n + self.l):
            pair_m2_reactions_row[f'R_{i+1}'] = pair_m2_opt['fluxes'][i]
        self.pair_m2opt_reactions_data.append(pair_m2_reactions_row)

        pair_m2_compounds = S_combined @ pair_m2_opt['fluxes']
        pair_m2_ext_part = pair_m2_compounds[:self.m]
        pair_m2_int1_part = pair_m2_compounds[self.m:self.m+self.p]
        pair_m2_int2_part = pair_m2_compounds[self.m+self.p:]
        
        pair_m2_ext_binary = self.flux_to_binary(pair_m2_ext_part, mode='consumption')
        pair_m2_int1_binary = self.flux_to_binary(pair_m2_int1_part, mode='activity')
        pair_m2_int2_binary = self.flux_to_binary(pair_m2_int2_part, mode='activity')
        pair_m2_compounds_binary = np.concatenate([pair_m2_ext_binary, pair_m2_int1_binary, pair_m2_int2_binary])
        pair_m2_compounds_row = {'env_id': env_id}
        for i in range(self.h + self.m + self.p):
            pair_m2_compounds_row[f'C_all_{i+1}'] = pair_m2_compounds_binary[i]
        self.pair_m2opt_compounds_data.append(pair_m2_compounds_row)
    
    def save_consolidated_csvs(self, output_dir, timestamp, model1_clean, model2_clean):
        files_saved = {}
        
        if self.environments_data:
            env_df = pd.DataFrame(self.environments_data)
            env_file = f"environments_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            env_path = os.path.join(output_dir, env_file)
            env_df.to_csv(env_path, index=False)
            files_saved['environments'] = env_file
    
        if self.summary_data:
            summary_df = pd.DataFrame(self.summary_data)
            summary_file = f"summary_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            summary_path = os.path.join(output_dir, summary_file)
            summary_df.to_csv(summary_path, index=False)
            files_saved['summary'] = summary_file
        
        if self.solo_m1_reactions_data:
            m1_reactions_df = pd.DataFrame(self.solo_m1_reactions_data)
            m1_reactions_file = f"solo_m1_reactions_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            m1_reactions_path = os.path.join(output_dir, m1_reactions_file)
            m1_reactions_df.to_csv(m1_reactions_path, index=False)
            files_saved['solo_m1_reactions'] = m1_reactions_file
        
        if self.solo_m1_ext_compounds_data:
            m1_ext_df = pd.DataFrame(self.solo_m1_ext_compounds_data)
            m1_ext_file = f"solo_m1_ext_compounds_binary_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            m1_ext_path = os.path.join(output_dir, m1_ext_file)
            m1_ext_df.to_csv(m1_ext_path, index=False)
            files_saved['solo_m1_ext_compounds'] = m1_ext_file
        
        if self.solo_m1_int_compounds_data:
            m1_int_df = pd.DataFrame(self.solo_m1_int_compounds_data)
            m1_int_file = f"solo_m1_int_compounds_binary_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            m1_int_path = os.path.join(output_dir, m1_int_file)
            m1_int_df.to_csv(m1_int_path, index=False)
            files_saved['solo_m1_int_compounds'] = m1_int_file
        
        if self.solo_m2_reactions_data:
            m2_reactions_df = pd.DataFrame(self.solo_m2_reactions_data)
            m2_reactions_file = f"solo_m2_reactions_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            m2_reactions_path = os.path.join(output_dir, m2_reactions_file)
            m2_reactions_df.to_csv(m2_reactions_path, index=False)
            files_saved['solo_m2_reactions'] = m2_reactions_file
        
        if self.solo_m2_ext_compounds_data:
            m2_ext_df = pd.DataFrame(self.solo_m2_ext_compounds_data)
            m2_ext_file = f"solo_m2_ext_compounds_binary_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            m2_ext_path = os.path.join(output_dir, m2_ext_file)
            m2_ext_df.to_csv(m2_ext_path, index=False)
            files_saved['solo_m2_ext_compounds'] = m2_ext_file
        
        if self.solo_m2_int_compounds_data:
            m2_int_df = pd.DataFrame(self.solo_m2_int_compounds_data)
            m2_int_file = f"solo_m2_int_compounds_binary_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            m2_int_path = os.path.join(output_dir, m2_int_file)
            m2_int_df.to_csv(m2_int_path, index=False)
            files_saved['solo_m2_int_compounds'] = m2_int_file
        
        if self.pair_m1opt_reactions_data:
            pair_m1_reactions_df = pd.DataFrame(self.pair_m1opt_reactions_data)
            pair_m1_reactions_file = f"pair_m1opt_reactions_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            pair_m1_reactions_path = os.path.join(output_dir, pair_m1_reactions_file)
            pair_m1_reactions_df.to_csv(pair_m1_reactions_path, index=False)
            files_saved['pair_m1opt_reactions'] = pair_m1_reactions_file
        
        if self.pair_m1opt_compounds_data:
            pair_m1_compounds_df = pd.DataFrame(self.pair_m1opt_compounds_data)
            pair_m1_compounds_file = f"pair_m1opt_compounds_binary_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            pair_m1_compounds_path = os.path.join(output_dir, pair_m1_compounds_file)
            pair_m1_compounds_df.to_csv(pair_m1_compounds_path, index=False)
            files_saved['pair_m1opt_compounds'] = pair_m1_compounds_file
        
        if self.pair_m2opt_reactions_data:
            pair_m2_reactions_df = pd.DataFrame(self.pair_m2opt_reactions_data)
            pair_m2_reactions_file = f"pair_m2opt_reactions_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            pair_m2_reactions_path = os.path.join(output_dir, pair_m2_reactions_file)
            pair_m2_reactions_df.to_csv(pair_m2_reactions_path, index=False)
            files_saved['pair_m2opt_reactions'] = pair_m2_reactions_file
        
        if self.pair_m2opt_compounds_data:
            pair_m2_compounds_df = pd.DataFrame(self.pair_m2opt_compounds_data)
            pair_m2_compounds_file = f"pair_m2opt_compounds_binary_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
            pair_m2_compounds_path = os.path.join(output_dir, pair_m2_compounds_file)
            pair_m2_compounds_df.to_csv(pair_m2_compounds_path, index=False)
            files_saved['pair_m2opt_compounds'] = pair_m2_compounds_file
        
        return files_saved
      
