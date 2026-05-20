import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix, csr_matrix, hstack


def load_model_simple(file_path):
    '''
    Load matlab model 
    '''
    data = loadmat(file_path)
    
    if 'MetabModel' in data:
        metab_model = data['MetabModel'][0, 0]
    elif 'metabolic_model' in data:
        metab_model = data['metabolic_model'][0, 0]
    else:
        for key in data.keys():
            if not key.startswith('__'):
                metab_model = data[key][0, 0]
                break
    keys = [
        "bmi", "rhs_ext_lb", "rhs_ext_ub", "rhs_int_lb", "rhs_int_ub",
        "S_ext", "S_int", "S_ext2int", "S_unmapped", "lb", "ub", "name"
    ]
    
    components = {}
    for i, key in enumerate(keys):
        if key.startswith("S_"):
            components[key] = csc_matrix(metab_model[i])
        elif key == "name":
            try:
                if hasattr(metab_model[i], '__iter__') and len(metab_model[i]) > 0:
                    components[key] = str(metab_model[i][0])
                else:
                    components[key] = str(metab_model[i])
            except:
                components[key] = f"model_{i}"
        else:
            components[key] = np.array(metab_model[i], dtype=np.float64)

    return components
  
