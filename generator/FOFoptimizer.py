import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csc_matrix, csr_matrix, hstack
from scipy.optimize import linprog, minimize
import cvxpy as cp
from sklearn.preprocessing import normalize
import warnings


class MetabolicOptimizer:
    '''
    Friend or Foe generator supports flexible metabolic optimization with multiple methods:
    - FBA: Standard Flux Balance Analysis (linear programming)
    - MOMA: Minimization of Metabolic Adjustment (quadratic programming)
    - pFBA: Parsimonious FBA (minimizes total flux)
    - L1_reg: L1 regularized FBA (promotes sparsity)
    - L2_reg: L2 regularized FBA (promotes small fluxes)
    '''
    def __init__(self, method='FBA', reference_fluxes=None, regularization_weight=0.001):
        self.method = method
        self.reference_fluxes = reference_fluxes
        self.regularization_weight = regularization_weight
        self.last_optimization_info = {}
    
    def optimize_microbe(self, microbe, env_rhslb):
        '''
        Optimize single microbe using specified method
        '''
        # Get model components
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
      
        if self.method == 'FBA':
            return self._solve_fba(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
        elif self.method == 'MOMA':
            return self._solve_moma(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
        elif self.method == 'pFBA':
            return self._solve_pfba(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
        elif self.method == 'L1_reg':
            return self._solve_l1_regularized(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
        elif self.method == 'L2_reg':
            return self._solve_l2_regularized(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
    
    def _solve_fba(self, n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense):
        '''
        Standard FBA
        '''
        c = np.zeros(n_vars)
        c[bmi] = -1.0
        
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,bounds=bounds, method='highs', options={'disp': False})
        
        if result.success:
            growth_rate = abs(result.fun)
            fluxes = result.x
            env_fluxes = S_ext_dense @ fluxes
            
            return {
                'growth_rate': float(growth_rate),
                'fluxes': fluxes,
                'env_fluxes': env_fluxes,
                'status': 'success',
                'optimization_info': {'method': 'FBA', 'solver_status': 'optimal'}
            }
        else:
            return {
                'growth_rate': 0.0,
                'fluxes': np.zeros(n_vars),
                'env_fluxes': np.zeros(len(b_ub)//2),
                'status': 'failed',
                'message': result.message
            }
    
    def _solve_moma(self, n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense):
        '''
        MOMA: Minimization of Metabolic Adjustment
        '''
        # Get reference fluxes using FBA
        ref_result = self._solve_fba(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
        if ref_result['status'] != 'success':
            return ref_result
        ref_fluxes = ref_result['fluxes']
        try:
            v = cp.Variable(n_vars)
            objective = cp.Minimize(cp.sum_squares(v - ref_fluxes))
            
            constraints = [
                A_ub @ v <= b_ub,
                v >= np.array([b[0] for b in bounds]),
                v <= np.array([b[1] for b in bounds]),
                v[bmi] >= 0.9 * ref_fluxes[bmi]
            ]
            
            if A_eq.size > 0:
                constraints.append(A_eq @ v == b_eq)
            
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if prob.status == cp.OPTIMAL:
                fluxes = v.value
                growth_rate = float(fluxes[bmi])
                env_fluxes = S_ext_dense @ fluxes
                
                return {
                    'growth_rate': float(growth_rate),
                    'fluxes': fluxes,
                    'env_fluxes': env_fluxes,
                    'status': 'success',
                    'optimization_info': {
                        'method': 'MOMA',
                        'solver_status': 'optimal',
                        'adjustment_distance': float(np.sum((fluxes - ref_fluxes)**2))
                    }
                }
            else:
                return ref_result  # FBA
        except Exception as e:
            return ref_result  # FBA
    
    def _solve_pfba(self, n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense):
        '''
        pFBA: Parsimonious FBA
        '''
        fba_result = self._solve_fba(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
        if fba_result['status'] != 'success':
            return fba_result
        max_growth = fba_result['growth_rate']
        
        try:
            v = cp.Variable(n_vars)
            objective = cp.Minimize(cp.sum(cp.abs(v)))
            
            constraints = [
                A_ub @ v <= b_ub,
                v >= np.array([b[0] for b in bounds]),
                v <= np.array([b[1] for b in bounds]),
                v[bmi] >= 0.999 * max_growth]
            if A_eq.size > 0:
                constraints.append(A_eq @ v == b_eq)
            
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if prob.status == cp.OPTIMAL:
                fluxes = v.value
                growth_rate = float(fluxes[bmi])
                env_fluxes = S_ext_dense @ fluxes
                
                return {
                    'growth_rate': float(growth_rate),
                    'fluxes': fluxes,
                    'env_fluxes': env_fluxes,
                    'status': 'success',
                    'optimization_info': {
                        'method': 'pFBA',
                        'solver_status': 'optimal',
                        'total_flux': float(np.sum(np.abs(fluxes)))
                    }
                }
            else:
                return fba_result
        except Exception as e:
            return fba_result
    
    def _solve_l1_regularized(self, n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense):
        """L1 FBA"""
        try:
            v = cp.Variable(n_vars)
            objective = cp.Maximize(v[bmi] - self.regularization_weight * cp.sum(cp.abs(v)))
            
            constraints = [
                A_ub @ v <= b_ub,
                v >= np.array([b[0] for b in bounds]),
                v <= np.array([b[1] for b in bounds])
            ]
            
            if A_eq.size > 0:
                constraints.append(A_eq @ v == b_eq)
            
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if prob.status == cp.OPTIMAL:
                fluxes = v.value
                growth_rate = float(fluxes[bmi])
                env_fluxes = S_ext_dense @ fluxes
                
                return {
                    'growth_rate': float(growth_rate),
                    'fluxes': fluxes,
                    'env_fluxes': env_fluxes,
                    'status': 'success',
                    'optimization_info': {
                        'method': 'L1_regularized',
                        'solver_status': 'optimal',
                        'l1_norm': float(np.sum(np.abs(fluxes)))
                    }
                }
            else:
                return self._solve_fba(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
        except Exception as e:
            return self._solve_fba(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
    
    def _solve_l2_regularized(self, n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense):
        '''
        L2 FBA
        '''
        try:
            v = cp.Variable(n_vars)
            objective = cp.Maximize(v[bmi] - self.regularization_weight * cp.sum_squares(v))
            
            constraints = [
                A_ub @ v <= b_ub,
                v >= np.array([b[0] for b in bounds]),
                v <= np.array([b[1] for b in bounds])
            ]
            
            if A_eq.size > 0:
                constraints.append(A_eq @ v == b_eq)
            
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if prob.status == cp.OPTIMAL:
                fluxes = v.value
                growth_rate = float(fluxes[bmi])
                env_fluxes = S_ext_dense @ fluxes
                
                return {
                    'growth_rate': float(growth_rate),
                    'fluxes': fluxes,
                    'env_fluxes': env_fluxes,
                    'status': 'success',
                    'optimization_info': {
                        'method': 'L2_regularized',
                        'solver_status': 'optimal',
                        'l2_norm': float(np.sum(fluxes**2))
                    }
                }
            else:
                return self._solve_fba(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
        except Exception as e:
            return self._solve_fba(n_vars, bmi, A_ub, b_ub, A_eq, b_eq, bounds, S_ext_dense)
