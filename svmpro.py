
from numpy import *
from time import sleep
from array555 import get_file

def clipAlpha(aj,H,L): #限制第二个Alpha的取值
    
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj


def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    #print 'kernelTrans'
    
    m,n = shape(X)
    
    
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            #print X[j,:].shape
            #print A.shape
            
            
            deltaRow = X[j,:] - A
            
            try:
                K[j] = deltaRow*deltaRow.T
            except:
                pass#K[j] = deltaRow.T*deltaRow
            #print K[j]
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
        print K
    else: raise NameError('We Have a Problem -- \
    That Kernel is not recognized')
    return K
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        print 'optStruct'
        self.X = dataMatIn
        self.labelMat = classLabels.T
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
            #print self.K
def calcEk(oS, k):
    
    
    
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    
    return Ek
        
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j=i #we want to select any J not equal to i
        while (j==i):
            j = int(random.uniform(0,oS.m))
        Ej = calcEk(oS, j)
    return j, Ej

def innerL(i, oS):
    print 'innerl'
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        Ek = calcEk(oS, j)
        oS.eCache[j] = [1,Ek] #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#通过第二个alpha值更新第一个alpha的值
        Ek = calcEk(oS, i)
        oS.eCache[i] = [1,Ek] #added this for the Ecache                    #更新第二个alpha值的误差值
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2  #更新阀值b
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    print 'smop'
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    print oS.eCache
    return oS.b,oS.alphas

def testSVM(kTup=('rbf', 10)):
    print 'testSVM'
    array_1,d1,array_2,d2 = get_file()
    datMat = array_1

    mytarget = d1.reshape(len(d1),1)
    
    labelMat = mytarget.transpose()
    b,alphas = smoP(array_1, mytarget,10, 0.0001, 10, kTup)
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    
    labelSV = labelMat[0][svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    print b
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        
        predict = predict.getA()
        predict = list(predict[0])

        try:
            if sign(predict[i])!=sign(labelMat[0][i]):errorCount += 1
        except:
            pass
    print "the training error rate is: %f" % (float(errorCount)/m)
   
    errorCount = 0

    datMat = array_2
    mytarget = d2.reshape(len(d2),1)
    labelMat = mytarget.transpose()
    m,n = shape(datMat)
    print b
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)

        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b

        predict = predict.getA()

        predict = list(predict[0])
       
        try:
            if sign(predict[i])!=sign(labelMat[0][i]):errorCount += 1
        except:
            pass 
    print "the test error rate is: %f" % (float(errorCount)/m) 
    
testSVM(kTup=('rbf', 10))

