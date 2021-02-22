import numpy as np
import shutil as sh

def energyIndicators(setting, experiment):
  datasetName = experiment.path.input+setting.id(sort=False).replace('step_energy', 'step_data')+'_energy.npy'

  data = np.load(datasetName)

  timeName = experiment.path.input+setting.id(hide=['source', 'part', 'typology', 'indicator'], sort=False).replace('step_partEnergy', 'step_data').replace('step_energy', 'step_data')+'_time.npy'


  # print(data.shape)

  ind = np.zeros((data.shape[0], 4))
  ti = np.zeros((data.shape[0], 1))
  for di, d in enumerate(data):
    ind[di, 0] = np.median(d[:, 0])
    ind[di, 1] = powerMean(d[:, 0])
    ind[di, 2] = np.median(d[:, 1])
    ind[di, 3] = powerMean(d[:, 1])

  outName = experiment.path.output+setting.id()
  np.save(outName+'_energy.npy', ind)

  sh.copyfile(timeName, outName+'_time.npy')

def powerMean(d):
  if np.mean(d)==0:
    return np.nan
  else:
    return 10*np.log10(np.mean(10 ** (d/10)))
