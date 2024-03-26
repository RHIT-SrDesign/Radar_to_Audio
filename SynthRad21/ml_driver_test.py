import ML_driver
from midas_tools import MidasFile

data_folder_path = "../SynthRad21/data/iq/angry_alpaca_sas.tmp"
model_path = "SynthRad21/temp/checkpoint/"
mf = MidasFile(data_folder_path)

data = mf.read_at_time(0, 0.1, reset_time=True)

p = ML_driver.characterize(model_path, data)

print(p)
