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

  experiment.metric.presence = ['mean']
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
    np.save(experiment.path.output+setting.id()+'_presence.npy', presence)
  if setting.step == 'part':
    # print(setting.source)
    if setting.sensor is not 'all':
      presence = getPresence(setting, experiment)
    else:
      presence = np.zeros(0)
      for k in range(len(experiment.factor.sensor)-1):
        presence = np.concatenate((presence, getPresence(setting.alternative('sensor', value=k), experiment)))

    # print(presence)
    presenceName = experiment.path.output+setting.id()+'_presence.npy'
    # print(presenceName)
    np.save(presenceName, presence)

  if setting.step in ['data', 'presence']:
    duration = time.time()-tic
    np.save(experiment.path.output+setting.id()+'_duration.npy', duration)

def getPresence(setting, experiment):
  presenceName = experiment.path.output+setting.id(hideFactor=['source', 'part']).replace('step_part', 'step_presence')+'_presence.npy'
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
      for b in range(presence.shape[1]):
        acc += presence[t, b, source]
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
