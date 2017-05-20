import neurolab as nl
import numpy as np

import os
import scipy.io as sio
import numpy as np
import pywt
import re
import random
    
def random_file(filelist):
        rfile = random.sample(filelist, 2*int(len(filelist)/3))
        lfile = set(filelist) - set(rfile)
        lfile = list(lfile)
        print 'random ok'
        return rfile,lfile
        
def get_file():
        N = re.compile('N')
        path = 'F:\data'
        a = list(os.walk(path))
        a = a[0]
        a = a[2]
        rfile,lfile = random_file(a)#lfile=left file rfile=select file
        array_1 = []
        array_2 = []
        n = 0
        d = []
        d2 = []
       
        for i in rfile: 
                x = i.split('.',1)
                if x[1]=='mat':
                        if bool(N.match(x[0]))==True:
                                i = str(i)
                                a = get_array(i)
                                if a is None:
                                        continue
                                else:
                                        array_1.append(a)
                                        d.append(1)
                        else:
                                i = str(i)
                                a = get_array(i)
                                if a is None:
                                        continue
                                else:
                                        array_1.append(a)
                                        d.append(-1)
                else:
                        pass
        for i in lfile: 
                x = i.split('.',1)
                if x[1]=='mat':
                        if bool(N.match(x[0]))==True:
                                i = str(i)
                                a = get_array(i)
                                if a is None:
                                        continue
                                else:
                                        array_2.append(a)
                                        d2.append(1)
                        else:
                                i = str(i)
                                a = get_array(i)
                                if a is None:
                                        continue
                                else:
                                        array_2.append(a)
                                        d2.append(-1)
                else:
                        pass
        array_1 = np.array(array_1)
        d = np.array(d)
        
        d2 = np.array(d2)
        
        return np.mat(array_1),d,np.mat(array_2),d2

def get_array(name):
        channel_num = 16
        array = []
        array_value = 0
        array_1 = []
        s = 0
        n = sio.loadmat(name)
        egMatrix = (n["dataStruct"][["data"]])[0][0][0]
        egMatrix = egMatrix.T
        egSamplingRate = list(n["dataStruct"][["iEEGsamplingRate"]])[0][0][0][0][0]
        egDataLength = list(n["dataStruct"][["nSamplesSegment"]])[0][0][0][0][0]
        egChannels = list(n["dataStruct"][["channelIndices"]])[0][0][0][0]
        for i in range(0,channel_num):   #(0,16)
            
                b = list(egMatrix[[i]])[0]
                array = []
                
                coeffs = pywt.wavedec(b,'db4', level=15)  
                a,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15 = coeffs

                for node in coeffs:    #calculate weight
                                for n in node:
                                        n = n*n
                                        s = s+n
                                array.append(s)
                                s = 0
                for y in array:
                                array_value = y*y+array_value
                array_value = np.sqrt(array_value)
                if array_value == 0:
                        return None
                else:
                        for z in array:
                                x = z/array_value
                                array_1.append(x)
        array_1 = np.array(array_1)
        return array_1
