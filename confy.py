import explanes as el
from pandas import DataFrame
import time
import numpy as np
from pathlib import Path
import os
import csv
import sys
import types
from datetime import datetime

if __name__ == "__main__":
  el.run.run()

def set(args):
  experiment = el.experiment.Experiment()
  experiment.project.name = 'confy'
  experiment.project.description = 'confy study using the cense sensor network'
  experiment.project.author = 'mathieu Lagrange'
  experiment.project.address = 'mathieu.lagrange@ls2n.fr'
  experiment.project.version = '0.1'

  experiment.path.code = '~/experiments/'+experiment.project.name+'/'
  experiment.path.input = '~/data/storage/cense/confy/data/'
  experiment.path.output = '~/drive/experiments/data/local/'+experiment.project.name+'/'
  experiment.setPath()

  experiment._factorFormatInReduce = 'long'
  experiment._metricFormatInReduce = 'short'
  experiment._factorFormatInReduceLength = 2
  experiment._metricFormatInReduceLength = 1


  experiment.host = ['pc-lagrange.irccyn.ec-nantes.fr']

  experiment.factor.step = ['data', 'presence']
  experiment.factor.month = ['january', 'march']
  experiment.factor.period = ['month', 'day', 'hour']
  experiment.factor.sensor = list(range(16))
  experiment.factor.default('period', 'month')

  experiment.metric.traffic = ['day', 'evening', 'night', 'full']
  experiment.metric.voice = ['day', 'evening', 'night', 'full']
  experiment.metric.bird = ['day', 'evening', 'night', 'full']
  experiment.metric.duration = ['mean']
  return experiment

def step(setting, experiment):
  tic = time.time()
  if setting.step is 'data':
    import prepareDataNpy
    prepareDataNpy.step(setting, experiment)
  if setting.step is 'presence':
    sys.path.append('../specialization')
    from inference import main
    config = types.SimpleNamespace()
    config.rnn = True
    config.modelName = 'train_scene_source_lorient'
    config.modelPath = experiment.path.output+'../censeDomainSpecialization/model/'
    config.datasetName = experiment.path.input+setting.alternative('step', 'data').id(sort=False)+'_spec.npy'
    config.outputPath = ''
    config.test = False
    # print(config.datasetName)
    # print(config)
    presence, timeOfPresence = main(config)
    # print(presence.shape)
    # print(timeOfPresence)

    timeVec = np.load(config.datasetName.replace('_spec.npy', '_time.npy'))
    sources = ['traffic', 'voice', 'bird']
    for iC in range(config.nClasses):
      predc = np.zeros((2, len(timeOfPresence)))
      for tpi, tp in enumerate(timeOfPresence):
        predc[0, tpi] = tp[iC]
        predc[1, tpi] = timeVec[tpi]
      np.save(experiment.path.output+setting.id()+'_'+sources[iC]+'.npy', predc)

  duration = time.time()-tic
  np.save(experiment.path.output+setting.id()+'_duration.npy', duration)

def selectData(data, period):
  acc = 0
  nbAcc = 0
  for t in range(data.shape[1]):
    h = datetime.utcfromtimestamp(data[1, t]/1000).hour
    if h>=period[0] and h<=period[1]:
      acc += data[0, t]
      nbAcc += 1
  return acc/nbAcc

def day(data):
  return np.mean(selectData(data, [7, 20]))

def evening(data):
  return np.mean(selectData(data, [20, 23]))

def night(data):
  return np.mean(selectData(data, [0, 7]))

def full(data):
  return np.mean(data[0, :])



# uncomment this method to fine tune display of metrics
# def display(experiment, settings):
#   (table, columns, header, nbFactorColumns) = experiment.metric.reduce(experiment.factor.mask(experiment.mask), experiment.path.output, factorDisplay=experiment._factorFormatInReduce, settingEncoding = experiment._settingEncoding, verbose=args.debug, reductionDirectiveModule=config)
#   # print(table)
#   # print(columns)
#   df = pd.DataFrame(table, columns=columns).fillna('')
#   df[columns[nbFactorColumns:]] = df[columns[nbFactorColumns:]].round(decimals=2)
#   if selectDisplay:
#     selector = [columns[i] for i in selectDisplay]
#     df = df[selector]
#   print(header)
#   print(df)
