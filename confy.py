import explanes as el
from pandas import DataFrame
import time
import numpy as np
from pathlib import Path
import os
import csv
import sys
import types
import datetime
import energy as en

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
  experiment.path.archive = '/tmp/'+experiment.project.name+'/'
  experiment.path.input = '~/data/storage/cense/confy/data/'
  experiment.path.output = '~/drive/experiments/data/local/'+experiment.project.name+'/'
  experiment.setPath()

  experiment._factorFormatInReduce = 'long'
  experiment._metricFormatInReduce = 'short'
  experiment._factorFormatInReduceLength = 2
  experiment._metricFormatInReduceLength = 1

  experiment.host = ['pc-lagrange.irccyn.ec-nantes.fr']

  experiment.factor.e1 = el.Factor()
  experiment.factor.e1.step = ['data']
  experiment.factor.e1.month = ['january', 'march']
  experiment.factor.e1.period = ['month', 'day', 'hour']
  experiment.factor.e1.sensor = list(range(16))
  experiment.factor.e1.default('period', 'month')

  experiment.factor.e2 = experiment.factor.e1.copy()
  experiment.factor.e2.step = ['presence']
  experiment.factor.e2.typology = ['tvb']

  experiment.factor.e3 = experiment.factor.e2.copy()
  experiment.factor.e3.step = ['part']
  experiment.factor.e3.sensor.append('all')
  experiment.factor.e3.source = ['traffic', 'voice', 'bird']
  experiment.factor.e3.part = ['day', 'evening', 'night', 'full']

  experiment.factor.e4 = experiment.factor.e2.copy()
  experiment.factor.e4.typology = ['tvbn']

  experiment.factor.e5 = experiment.factor.e4.copy()
  experiment.factor.e5.step = ['part']
  experiment.factor.e5.sensor.append('all')
  experiment.factor.e5.source = ['traffic', 'voice', 'bird', 'background']
  experiment.factor.e5.part = ['day', 'evening', 'night', 'full']

  experiment.factor.e6 = experiment.factor.e2.copy()
  experiment.factor.e6.typology = ['cmtvbsn']

  experiment.factor.e7 = experiment.factor.e6.copy()
  experiment.factor.e7.step = ['part']
  experiment.factor.e7.sensor.append('all')
  experiment.factor.e7.source = ['car', 'motorbike', 'truck', 'voice', 'birds', 'seagulls', 'background']
  experiment.factor.e7.part = ['day', 'evening', 'night', 'full']

  experiment.factor.e8 = experiment.factor.e1.copy()
  experiment.factor.e8.step = ['energy']

  experiment.factor.e9 = experiment.factor.e8.copy()
  experiment.factor.e9.step = ['partEnergy']
  experiment.factor.e9.sensor.append('all')
  experiment.factor.e9.source = ['laeqMedian', 'laeqMean', 'leqMedian', 'leqMean']
  experiment.factor.e9.part = ['day', 'evening', 'night', 'full']


  experiment.metric.presence = ['mean%']
  experiment.metric.energy = ['mean']

  # experiment.metric.presence = ['mean%', 'std%']
  # experiment.metric.timeOfPresence = ['mean%', 'std%']
  # experiment.metric.duration = ['mean']
  return experiment

def step(setting, experiment):
  tic = time.time()
  if setting.step == 'data':
    import prepareDataNpy
    prepareDataNpy.step(setting, experiment)
  if setting.step == 'presence' and setting.sensor != 'all':
    sys.path.append('../specialization')
    from inference import main
    config = types.SimpleNamespace()
    config.rnn = True
    config.sensorData = True
    config.modelName = 'train_scene_source_lorient'
    config.modelPath = experiment.path.output+'../specialization/model_'+setting.typology+'/'
    config.datasetName = experiment.path.input+setting.id(sort=False).replace('step_presence', 'step_data').replace('_typology_'+setting.typology, '')+'_spec.npy'
    config.outputPath = ''
    config.test = False
    config.classes = list(setting.typology)
    config.debug = experiment.status.debug
    presence, timeOfPresence = main(config)
    if experiment.status.debug:
      print(presence.shape)
      print(timeOfPresence.shape)

    np.save(experiment.path.output+setting.id()+'_presence.npy', presence)
    # np.save(experiment.path.output+setting.id()+'_timeOfPresence.npy', timeOfPresence)
  if setting.step == 'energy':
    en.energyIndicators(setting, experiment)
  if setting.step == 'part':
    # print(setting.source)
    if setting.sensor != 'all':
      presence = getData(setting, experiment)
      # timeOfPresence = getData(setting, experiment, type='timeOfPresence')
    else:
      presence = np.zeros(0)
      timeOfPresence = np.zeros(0)
      for k in range(len(experiment.factor.sensor)-1):
        presence = np.concatenate((presence, getData(setting.replace('sensor', value=k), experiment)))
        timeOfPresence = np.concatenate((timeOfPresence, getData(setting.replace('sensor', value=k), experiment, 'timeOfPresence')))

    # print(presence)
    presenceName = experiment.path.output+setting.id()+'_presence.npy'
    # print(presence.shape)
    np.save(presenceName, presence)
    timeOfPresenceName = experiment.path.output+setting.id()+'_timeOfPresence.npy'
    # print(timeOfPresence.shape)
    np.save(timeOfPresenceName, timeOfPresence)
  if setting.step == 'partEnergy':
    # print(setting.source)
    if setting.sensor != 'all':
      energy = getData(setting, experiment, type='energy')
    else:
      energy = np.zeros(0)
      for k in range(len(experiment.factor.sensor)-1):
        energy = np.concatenate((energy, getData(setting.replace('sensor', value=k), experiment, type='energy')))

    energyName = experiment.path.output+setting.id()+'_energy.npy'
    # print(energy.shape)
    np.save(energyName, energy)

  if setting.step in ['data', 'presence']:
    duration = time.time()-tic
    np.save(experiment.path.output+setting.id()+'_duration.npy', duration)

def getData(setting, experiment, type='presence'):

  pathName = experiment.path.output+setting.id(hide=['source', 'part', 'indicator']).replace('step_partEnergy', 'step_'+type)+'_'+type+'.npy'

  if experiment.status.debug:
    print(pathName)
  data = np.load(pathName)
  # print(data.shape)
  timeName = experiment.path.input+setting.id(hide=['source', 'part', 'typology', 'indicator'], sort=False).replace('step_partEnergy', 'step_data').replace('step_part', 'step_data')+'_time.npy'
  timeVec = np.load(timeName)

  if setting.part == 'day':
    period = [7, 20]
  if setting.part == 'evening':
    period = [20, 23]
  if setting.part == 'night':
    period = [0, 7]
  if setting.part == 'full':
    period = [0, 24]

  source = experiment.factor.source.index(setting.source)
  # print(source)

  return selectData(data, timeVec, period, source)

def selectData(presence, time, period, source):
  acc = 0
  nbAcc = 0
  # print(presence.shape)
  # print(time.shape)
  a = []
  ph=-1
  for t in range(presence.shape[0]):
    h = datetime.datetime.utcfromtimestamp(time[t]/1000).hour
    # print(h)
    if h>=period[0] and h<=period[1]:
      run = True
      # print(presence.shape)
      if presence.ndim>2:
        for b in range(presence.shape[1]):
          acc += presence[t, b, source]
          nbAcc += 1
      else:
        acc += presence[t, source]
        nbAcc += 1
    if ph>h:
      if nbAcc:
        a.append(acc/nbAcc)
      acc = 0
      nbAcc = 0
    else:
      ph=h

  if len(a)==0:
    a.append(0)
  return np.array(a)
