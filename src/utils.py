import numpy as np

def encode_label(label):
	assert(len(label) == 4)
	alphabets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
	'P','Q','R','S','T','U','V','W','X','Y','Z']

	section = np.zeros((1,8))
	clas = np.zeros((1,100))
	subsection = np.zeros((1,26))
	section[0, alphabets.index(label[0])] = 1
	clas[0, int(label[1:3])] = 1
	subsection[0, alphabets.index(label[0])] = 1
	return np.squeeze(np.hstack([section, clas, subsection]))