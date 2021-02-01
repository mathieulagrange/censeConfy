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

  experiment.factor.e1 = el.Factor()
  experiment.factor.e1.step = ['data', 'presence']
  experiment.factor.e1.month = ['january', 'march']
  experiment.factor.e1.period = ['month', 'day', 'hour']
  experiment.factor.e1.sensor = list(range(16))
  experiment.factor.e1.default('period', 'month')

  experiment.factor.e2 = experiment.factor.e1.copy()
  experiment.factor.e2.step.append('part')
  experiment.factor.e2.sensor.append('all')
  experiment.factor.e2.source = ['traffic', 'voice', 'bird']
  experiment.factor.e2.part = ['day', 'evening', 'night', 'full']

  experiment.metric.presence = ['mean%', 'std%']
  experiment.metric.timeOfPresence = ['mean%', 'std%']
  experiment.metric.duration = ['mean']
  return experiment

def step(setting, experiment):
  tic = time.time()
  if setting.step == 'data':
    import prepareDataNpy
    prepareDataNpy.step(setting, experiment)
  if setting.step is 'presence':
    sys.path.append('../specialization')
    from inference import main
    config = types.SimpleNamespace()
    config.rnn = True
    config.sensorData = True
    config.modelName = 'train_scene_source_lorient'
    config.modelPath = experiment.path.output+'../censeDomainSpecialization/model/'
    config.datasetName = experiment.path.input+setting.alternative('step', 'data').id(sort=False)+'_spec.npy'
    config.outputPath = ''
    config.test = False
    # print(config.datasetName)
    # print(config)
    presence, timeOfPresence = main(config)
    # print(presence.shape)
    # print(timeOfPresence.shape)

    np.save(experiment.path.output+setting.id()+'_presence.npy', presence)
    np.save(experiment.path.output+setting.id()+'_timeOfPresence.npy', timeOfPresence)
  if setting.step == 'part':
    # print(setting.source)
    if setting.sensor is not 'all':
      presence = getData(setting, experiment)
      timeOfPresence = getData(setting, experiment, type='timeOfPresence')
    else:
      presence = np.zeros(0)
      timeOfPresence = np.zeros(0)
      for k in range(len(experiment.factor.sensor)-1):
        presence = np.concatenate((presence, getData(setting.alternative('sensor', value=k), experiment)))
        timeOfPresence = np.concatenate((timeOfPresence, getData(setting.alternative('sensor', value=k), experiment, 'timeOfPresence')))

    # print(presence)
    presenceName = experiment.path.output+setting.id()+'_presence.npy'
    # print(presence.shape)
    np.save(presenceName, presence)
    timeOfPresenceName = experiment.path.output+setting.id()+'_timeOfPresence.npy'
    # print(timeOfPresence.shape)
    np.save(timeOfPresenceName, timeOfPresence)

  if setting.step in ['data', 'presence']:
    duration = time.time()-tic
    np.save(experiment.path.output+setting.id()+'_duration.npy', duration)

def getData(setting, experiment, type='presence'):
  presenceName = experiment.path.output+setting.id(hideFactor=['source', 'part']).replace('step_part', 'step_presence')+'_'+type+'.npy'
  # print(presenceName)
  presence = np.load(presenceName)

  timeName = experiment.path.input+setting.id(hideFactor=['source', 'part'], sort=False).replace('step_part', 'step_data')+'_time.npy'
  # print(timeName)
  timeVec = np.load(timeName)
  if setting.part == 'day':
    period = [7, 20]
  if setting.part == 'evening':
    period = [20, 23]
  if setting.part == 'night':
    period = [0, 7]
  if setting.part == 'full':
    period = [0, 24]
  if setting.source == 'traffic':
    source = 0
  if setting.source == 'voice':
    source = 1
  if setting.source == 'bird':
    source = 2

  return selectData(presence, timeVec, period, source)

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
