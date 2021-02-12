import numpy as np

def energyIndicators(setting, experiment):
  datasetName = experiment.path.input+setting.id(sort=False).replace('step_energy', 'step_data')+'_energy.npy'

  data = np.load(datasetName)

  ind = np.zeros((data.shape[0], 4))
  for di, d in enumerate(data):
    ind[di, 0] = np.median(d[:, 0])
    ind[di, 1] = powerMean(d[:, 0])
    ind[di, 2] = np.median(d[:, 1])
    ind[di, 3] = powerMean(d[:, 1])

  outName = experiment.path.output+setting.id()+'_energy.npy'
  np.save(outName, ind)

def powerMean(d):
  if np.mean(d)==0:
    return 0
  else:
    return 10*np.log10(np.mean(10*(d/10)))
