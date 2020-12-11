import numpy as np
import os
import glob
import csv
import shutil
import tqdm as tq

from utils import *

# class Sensor(tb.IsDescription):
#   number = tb.Int32Col()
#   id = tb.StringCol(100)
#   lat = tb.Float64Col()
#   lon = tb.Float64Col()

rootPath = os.path.expanduser('~/data/storage/cense/confy/')
inputPath = rootPath+'raw/'
dataPath = rootPath+'data/'

localPath = os.path.expanduser('~/drive/experiments/data/local/censeConfy/')

def step(setting, experiment):
  nbFrequencyBands = 29
  nbVec = 351

  sensorInfo = sensor_list('sensorList.csv')
  monthPath = ['01_02_2020', '04_05_2020']

  if setting.month is 'january':
    monthPath = '01_02_2020'
  if setting.month is 'march':
    monthPath = '04_05_2020'

  if setting.period in ['day', 'hour']:
    dayLimit = 2
  else:
    dayLimit = 33
  if setting.period == 'hour':
    hourLimit = 2
  else:
    hourLimit = 25

  dataId = setting.id(sort=False)
  # print(dataPath+dataId)
  if not os.path.exists(dataPath):
    os.makedirs(dataPath)
  #for setting.sensor, s in enumerate(sensorInfo): #tq.tqdm(enumerate(sensorInfo), total=len(sensorInfo)):
  # print('Sensor '+str(setting.sensor)+' / '+str(len(sensorInfo)))

  fileNames = []
  for year in [2019, 2020]:
    for month in range(13):
      for day in range(dayLimit):
        for hour in range(hourLimit):
          f = inputPath+monthPath+'/'+sensorInfo[setting.sensor]["sID"]+'/'+str(year)+'/'+str(month)+'/'+str(day)+'/'+str(hour)+'.zip'
          # print(f)
          if os.path.exists(f):
            fileNames.append(f)
  fCount = 0
  arrayTime = np.zeros((len(fileNames)*6))
  arraySpec = np.zeros((len(fileNames)*6, nbVec, nbFrequencyBands))
  # print(fileNames)
  for inputFileName in tq.tqdm(fileNames, total=len(fileNames)):
    shutil.copy(inputFileName, '/tmp/confy.zip')
    os.system('unzip -qq -o -d /tmp /tmp/confy.zip')
    csvFileName = os.path.basename(inputFileName).replace('zip', 'csv')
    with open('/tmp/'+csvFileName, 'r') as csvfileID:
      reader = csv.reader(csvfileID, delimiter=',')

      data = np.zeros((nbVec, 32))
      rCount = 0
      for r, row in enumerate(reader):
        if r%4800<nbVec: # every 10 minutes, for 45 seconds
            #print(rCount)
          data[rCount, :] = np.array([float(s) for s in row])
          rCount +=1
        elif r%4800==nbVec:
          rCount = 0
          arrayTime[fCount] = data[0, 0]
          arraySpec[fCount, :, :] = data[:, 3:][None]
          fCount += 1
            #print(arraySpec.shape)
      os.remove('/tmp/'+csvFileName)
    os.remove('/tmp/confy.zip')
  np.save(dataPath+dataId+'_time.npy', arrayTime)
  np.save(dataPath+dataId+'_spec.npy', arraySpec)
