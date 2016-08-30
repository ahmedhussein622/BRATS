

import keras
print(keras.__version__)
from Patient import Patient

Patient.find_patients()
for p in Patient.patients_list :
    p.start_iteration()
    p.stop_iteration()

