import numpy as np
import tables as tb
import os
import glob
import csv
import shutil
import tqdm as tq
import argparse
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,9]

from utils import *

class Config:
    pass

class Sensor(tb.IsDescription):
  number = tb.Int32Col()
  id = tb.StringCol(100)
  lat = tb.Float64Col()
  lon = tb.Float64Col()

config = Config()
config.nbFrequencyBands = 29
config.nbVec = 351

config.rootPath = os.path.expanduser('~/data/storage/cense/confy/')
config.inputPath = config.rootPath+'raw/'
config.dataPath = config.rootPath+'data/'

config.sensorInfo = sensor_list('sensorList.csv')

config.month = ['january', 'march']
config.monthPath = ['01_02_2020', '04_05_2020']

config.localPath = os.path.expanduser('~/drive/experiments/data/local/censeConfy/')

def generateDataset(config):
  tmpFileName = config.dataPath+'tmpDa'
  dataId = 'censeConfy'+config.month[config.monthType]

  if config.oneDay:
    dayLimit = 2
  else:
    dayLimit = 33
  if config.oneHour:
    hourLimit = 2
  else:
    hourLimit = 25


  if config.dryRun:
    dataId += 'DryRun'
  if config.oneDay:
    dataId += 'OneDay'
  if config.oneHour:
    dataId += 'OneHour'
  print(dataId)
  if not os.path.exists(config.dataPath+dataId):
      os.makedirs(config.dataPath+dataId)
  for sCount, s in enumerate(config.sensorInfo): #tq.tqdm(enumerate(sensorInfo), total=len(sensorInfo)):
    print('Sensor '+str(sCount)+' / '+str(len(config.sensorInfo)))

    if config.dryRun:
      data = np.zeros((10, 32))
      arrayTime=data[:, 0]
      arraySpec=data[:, 3:]
    else:
      fileNames = []
      for year in [2019, 2020]:
        for month in range(13):
          for day in range(dayLimit):
            for hour in range(hourLimit):
              f = config.inputPath+config.monthPath[config.monthType]+'/'+s["sID"]+'/'+str(year)+'/'+str(month)+'/'+str(day)+'/'+str(hour)+'.zip'
              if os.path.exists(f):
                fileNames.append(f)
      fCount = 0
      arrayTime = np.zeros((len(fileNames)*6))
      arraySpec = np.zeros((len(fileNames)*6, config.nbVec, config.nbFrequencyBands))

      for inputFileName in tq.tqdm(fileNames, total=len(fileNames)):
        shutil.copy(inputFileName, tmpFileName+'.zip')
        os.system('unzip -qq -d /tmp '+tmpFileName+'.zip')
        csvFileName = os.path.basename(inputFileName).replace('zip', 'csv')
        with open('/tmp/'+csvFileName, 'r') as csvfileID:
          reader = csv.reader(csvfileID, delimiter=',')

          data = np.zeros((config.nbVec, 32))
          rCount = 0
          for r, row in enumerate(reader):
            if r%4800<config.nbVec: # every 10 minutes, for 45 seconds
                #print(rCount)
              data[rCount, :] = np.array([float(s) for s in row])
              rCount +=1
            elif r%4800==config.nbVec:
              rCount = 0
              arrayTime[fCount] = data[0, 0]
              arraySpec[fCount, :, :] = data[:, 3:][None]
              fCount += 1
                #print(arraySpec.shape)
          os.remove('/tmp/'+csvFileName)
        os.remove(tmpFileName+'.zip')
    np.save(config.dataPath+dataId+'/'+dataId+'_sensor_'+str(sCount)+'_time.npy', arrayTime)
    np.save(config.dataPath+dataId+'/'+dataId+'_sensor_'+str(sCount)+'_spec.npy', arraySpec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('-D', action='store_true', help='OneDay')
    parser.add_argument('-d', action='store_true', help='debug')
    parser.add_argument('-H', action='store_true', help='OneHour')
    parser.add_argument('-C', action='store_true', help='confy')

    args = parser.parse_args()
    config.oneDay = args.D
    config.oneHour = args.H
    config.monthType = args.C
    config.dryRun = args.d

    generateDataset(config)
