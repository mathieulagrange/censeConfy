import numpy as np
import os
import csv
from pyproj import Proj, transform
import math
import dateutil.parser
from datetime import datetime

def gps2merc(lat, lon):
    r_major = 6378137.000
    x = r_major * math.radians(lon)
    scale = x/lon
    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 +
    lat * (math.pi/180.0)/2.0)) * scale
    return (x, y)

def sensor_list(sListFile):
    projGPS = Proj('epsg:4326')
    projWM = Proj('epsg:3857')
    with open(sListFile, 'r') as csvfileID:
        reader = csv.reader(csvfileID, delimiter=',')
        rowCount = sum(1 for row in csvfileID)
        # print('Found {} sensors.'.format(rowCount))
        sensorInfo = []
        csvfileID.seek(0)
        for row in reader:
            latWM, lonWM = gps2merc(float(row[1]), float(row[2]))
            sensorInfo.append(dict(sID=row[0], latGPS=row[2], lonGPS=row[1], latWM=latWM, lonWM=lonWM))
    return sensorInfo

def list_csv_files(sID):
    location = os.path.join('sensor_data', sID)
    csv_files = []
    csv_tstart = []
    csv_tend = []
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in sorted(filenames) if f.endswith((".csv"))]:
            try:
                csv_tstart.append(dateutil.parser.parse(filename[:27]))
                csv_tend.append(dateutil.parser.parse(filename[28:-4]))
                csv_files.append(filename)
            except dateutil.parser._parser.ParserError:
                #print('Ignoring {}'.format(filename))
                pass

    if len(csv_files) == 0:
        print('Error: Found no csv files in ' + location)
    return csv_files, csv_tstart, csv_tend

def sensor_stats(sID, tStart, tInt):
    # Find all files with right time
    csv_files, csv_tstart, csv_tend = list_csv_files(sID)
    cFiles = []
    for iF, f in enumerate(csv_files):
        if not ((csv_tstart[iF] > tStart+tInt) or (csv_tend[iF] < tStart)):
            cFiles.append(f)
    # First pass in files to determine shape of np arrays
    nRows = 0
    for iF, f in enumerate(cFiles):
        with open(os.path.join('sensor_data', sID, f), 'r') as fID:
            reader = csv.reader(fID, delimiter=',')
            for iRow, row in enumerate(reader):
                if iRow >=1:
                    tRow = dateutil.parser.parse(row[2])
                    if tRow >= tStart and tRow <= tStart+tInt:
                        nRows += 1

    # Second pass, get the data
    sData = dict(sID=sID, tStart=tStart, tInt=tInt, Leq=np.zeros((nRows, 1)), Laeq=np.zeros((nRows, 1)), spec=np.zeros((nRows, 29)))
    nRow_current = 0
    for iF, f in enumerate(cFiles):
        with open(os.path.join('sensor_data', sID, f), 'r') as fID:
            reader = csv.reader(fID, delimiter=',')
            for iRow, row in enumerate(reader):
                if iRow >=1:
                    tRow = dateutil.parser.parse(row[2])
                    if tRow >= tStart and tRow <= tStart+tInt:
                        sData['Leq'][nRow_current] = float(row[3])
                        sData['Laeq'][nRow_current] = float(row[4])
                        sData['spec'][nRow_current, :] = [float(s) for s in row[5:]]
                        nRow_current += 1
    LeqAvg = np.mean(10*np.log10(np.power(10, sData['Leq']/10)), axis=0)
    LaeqAvg = np.mean(10*np.log10(np.power(10, sData['Laeq']/10)), axis=0)
    specAvg = np.mean(10*np.log10(np.power(10, sData['spec']/10)), axis=0)
    spc = spectral_centroid(specAvg)
    return LeqAvg, LaeqAvg, specAvg, spc


# TODO
def time_presence_cnn(X):
    pass
    # TODO move to single init


def test_simple(X, batch_size=1):
    model = ActPredCNN()
    model = torch.load('model_2019-02-21_16-12-30.pt')
    nEx = int(np.floor(X.shape[0]/8))
    # TODO Batches
    Xb = torch.zeros((batch_size, 1, 8, 29))
    T
    for iEx in range(nEx):
        Xb[0, 0, :, :] = torch.from_numpy(X[iEx*8:(iEx+1)*8, :])
        Xb = Variable(Xb.type(torch.FloatTensor)).cuda()
        #Xb = Xb.cuda()
        o = model(Xb)
        o = F.sigmoid(o)
        # TODO recup tpres
        # TODO Normalization input (learned on electrical db?)


    '''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
    t_pres_preds = []
    for current_batch, x in enumerate(tqdm(self.dataloader, desc='Test')):
        x = Variable(x.type(torch.FloatTensor))
        x = x.cuda()

        o = self.model(x)
        o = F.sigmoid(o)

        pres_pred = o.round().cpu().data
        t_pres_pred = torch.mean(pres_pred, dim=0)

        t_pres_preds.append(t_pres_pred)
        #print(t_pres_pred)
    '''
