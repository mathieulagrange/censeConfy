import explanes as el
from pandas import DataFrame
import time
import numpy as np
from pathlib import Path
import os

if __name__ == "__main__":
  el.experiment.run()

# use case where:
#   - the results are stored on disk using npy files
#   - one factor affects the size of the results vectors
#   - the metrics does not operate on the same data, resulting on result vectors with different sizes per metric

def set(args):
  experiment = el.experiment.Experiment()
  experiment.project.name = 'censeConfy'
  experiment.project.description = 'confy study using the cense sensor network'
  experiment.project.author = 'mathieu Lagrange'
  experiment.project.address = 'mathieu.lagrange@ls2n.fr'
  experiment.project.version = '0.1'

  experiment.path.input = '~/data/storage/cense/confy/data/'
  experiment.path.predict = '~/data/experiments/'+experiment.project.name+'/predict/'
  experiment.path.aggregate = '~/data/experiments/'+experiment.project.name+'/aggregate/'
  experiment.path.code = '~/experiments/censeConfy/'
  # experiment.path.tmp = '/tmp/'+experiment.runId
  experiment.setPath()

  experiment._archivePath = '/tmp'

  experiment.host = ['pc-lagrange.irccyn.ec-nantes.fr']

  experiment.factor.task = ['predict', 'aggregate']
  experiment.factor.month = ['january', 'march']
  experiment.factor.reduce = ['none', 'OneDay', 'OneDayOneHour']
  experiment.factor.sensor = list(range(16))
#  experiment.factor.period = ['none', 'day', 'evening', 'night']

  experiment.metric.t = ['mean', 'std']
  experiment.metric.v = ['mean', 'std']
  experiment.metric.b = ['mean', 'std']
  experiment.metric.duration = ['mean']
  return experiment

def step(setting, experiment):
  tic = time.time()
  if setting.task is 'predict':
    dataFileName = 'censeConfy'+setting.month
    if setting.reduce is not 'none':
      dataFileName += setting.reduce
    dataFileName = experiment.path.input+dataFileName+'/'+dataFileName+'_sensor_'+str(setting.sensor)+'_spec.npy'
    print(dataFileName)
    command = 'cd ../censeDomainSpecialization && python3 inference.py --dataset TVBCense_dev -rnn --datasetName '+dataFileName+' --outputPath  '+experiment.path.predict+' --pretrained vec -finetune'+' --outputName  '+setting.id()
    print(command)
    os.system(command)

  baseFileName = setting.id()
  duration = time.time()-tic
  np.save(os.path.expanduser(experiment.path.predict+baseFileName+'_duration.npy'), duration)

# uncomment this method to fine tune display of metrics
def display(experiment, settings):
  (data, desc, header)  = experiment.metric.get('mae', settings, experiment.path.output, settingEncoding = experiment._settingEncoding)

  print(header)
  print(desc)
  print(len(data))
