
import numpy as np
import tables as tb
import os
import glob
import csv
import shutil
import tqdm
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,9]

from utils import *

nbFrequencyBands = 29

rootPath = '/home/lagrange/data/storage/cense/confy/'
inputPath = rootPath+'raw/'
dataPath = rootPath+'data/'

sListFile = 'sensor_list.csv'
sensorInfo = sensor_list(sListFile)

datasetFileName = 'censeConfy.h5'

"""# Generate dataset"""

class Sensor(tb.IsDescription):
  number = tb.Int32Col()
  id = tb.StringCol(100)
  lat = tb.Float64Col()
  lon = tb.Float64Col()


data = np.zeros((dataFileSize, nbFrequencyBands))
dataTime = np.zeros((dataFileSize, 1))

dryRun = False
reDo = True

if 'f' in locals():
  f.close()

if reDo :
  fileName = dataPath+datasetFileName
  if dryRun:
    fileName = fileName.replace('.h5', 'DryRun.h5')
  f = tb.open_file(fileName, mode='w')
  f.create_table('/', 'sensor', Sensor, 'Sensor information')
  time = f.create_group('/', 'time', 'time expressed in epoch')
  spectrum = f.create_group('/', 'spectrum', 'spectral data third octave bands fast (125ms)')
  for sCount, s in tqdm(enumerate(sensorInfo), total=len(sensorInfo)):
    sensor = f.root.sensor.row
    sensor['id'] = s["sID"]
    sensor['lat'] = s["latGPS"]
    sensor['lon'] = s["lonGPS"]
    sensor['number'] = sCount
    sensor.append()
    arrayTime = f.create_earray(time, 'sensor'+str(sCount), tb.Float64Atom(), (0,))
    arraySpec = f.create_earray(spectrum, 'sensor'+str(sCount), tb.Float64Atom(), (0, nbFrequencyBands))
    #print('----- '+s["sID"]+' -----')
    if dryRun:
      if np.random.randint(1):
        data = np.zeros((10, 32))
        arrayTime.append(data[:, 0])
        arraySpec.append(data[:, 3:])
    else:
      fileNames = []
      for year in [2019, 2020]:
        for month in range(13):
          for day in range(33):
            for hour in range(25):
              fileName = dataPath+'fetch/'+s["sID"]+'/'+str(year)+'/'+str(month)+'/'+str(day)+'/'+str(hour)+'.zip'
              if os.path.exists(fileName):
                fileNames.append(fileName)
      # print(len(fileNames))

      for fileName in fileNames:
        shutil.copy(fileName, tmpFileName+'.zip')
        os.system('unzip -d /tmp '+tmpFileName+'.zip')
        csvFileName = os.path.basename(fileName).replace('zip', 'csv')
        with open('/tmp/'+csvFileName, 'r') as csvfileID:
          reader = csv.reader(csvfileID, delimiter=',')
          nbVec = (sum(1 for row in reader))
        with open('/tmp/'+csvFileName, 'r') as csvfileID:
          reader = csv.reader(csvfileID, delimiter=',')
          data = np.zeros((nbVec, 32))
          rCount = 0
          for row in reader:
            data[rCount, :] = [float(s) for s in row]
            rCount +=1
          arrayTime.append(data[:, 0])
          arraySpec.append(data[:, 3:])

          os.remove('/tmp/'+csvFileName)
        os.remove(tmpFileName+'.zip')
  f.root.sensor.flush()
  f.close()
