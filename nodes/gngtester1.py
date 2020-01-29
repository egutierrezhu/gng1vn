from __future__ import division 
import numpy as np 
import scipy.io.wavfile as wav

from features import mfcc
from numpy import linalg as LA

class Node():

    def __init__(self, name = None, neighbors = None, weight = None, label = None,
            age = None, error = None):


        if neighbors is None:
            neighbors = []
        if age is None:
            age= []
        if error is None:
            error= []
        if weight is None:
            weight= []

        self.name = name
        self.neighbors = neighbors
        self.weight = weight
        self.label = label
        self.age = age
        self.error = error

class TestingNetwork:

    def __init__(self,node_list):


        self.node_list = node_list
        self.labels = []
        self._count = []

        # online cumulative mistake rate
        self.error_count = 0

    def _distance(self, v1, v2):

        # Returns the euclidean distance between v1 and v2

        assert v1.shape == v2.shape, "V1 and V2 must be of same shape"

        return np.sqrt(np.sum(np.square(v1 - v2)))

    def test_single(self, data):

        """
        Returns the label obtained by the network for a single data sample
        """
        
        menor_dist = np.inf
        
        for node in self.node_list:
            dist = self._distance(node.weight, data)

            if dist < menor_dist:
                menor_dist = dist
                label = int(node.label)
                best_node = node

        return label

def testInit(filename_som1):
	#Setup Neural Network
        fsom = open(filename_som1, "rb")
	node_list  = np.load(fsom)
	testNet = TestingNetwork(node_list)
	return testNet

def extractFeature(soundfile):
	#Get MFCC Feature Array
	(rate,sig) = wav.read(soundfile)
	duration = len(sig)/rate;	
	mfcc_feat = mfcc(sig,rate,winlen=duration/20,winstep=duration/20)
	s = mfcc_feat[:20]
	st = []
	for elem in s:
		st.extend(elem)
	st /= np.max(np.abs(st),axis=0)
	inputArray = np.array([st])
	return inputArray

def feedToNetwork(words,inputArray,testNet):
	# Input MFCC Array to Network
	outputLabel = testNet.test_single(inputArray[0])

        # Label to index word

        indexMax = outputLabel-1
			
	# Mapping each index to their corresponding meaning

        return words[indexMax]

if __name__ == "__main__":

        words = ['backward','forward','go','left','right','stop']

        filename_som1 = "maps/nodes_cmd_6words.npy"
        filename_test = "test_files/test.wav"

        print ("Testing SGNG: " + filename_test)
        
        testNet = testInit(filename_som1)

        inputArray = extractFeature(filename_test)      
        
        testStr = feedToNetwork(words,inputArray,testNet)

        print (testStr)





	


	
		


