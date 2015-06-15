import numpy as np
import scipy.io
import scipy.signal

def hanningWindow(nx, ny, nPixX, nPixY):
	"""
	Return a Hanning window in 2D
	
	Args:
	    nx (TYPE): number of pixels in x-direction of mask
	    ny (TYPE): number of pixels in y-direction of mask
	    nPixX (TYPE): number of pixels in x-direction of the transition
	    nPixY (TYPE): number of pixels in y-direction of the transition
	
	Returns:
		 real: 2D apodization mask
		
	"""					
	winX = np.hanning(nPixX)
	winY = np.hanning(nPixY)

	winOutX = np.ones(nx)
	winOutX[0:nPixX/2] = winX[0:nPixX/2]
	winOutX[-nPixX/2:] = winX[-nPixX/2:]

	winOutY = np.ones(ny)
	winOutY[0:nPixY/2] = winY[0:nPixY/2]
	winOutY[-nPixY/2:] = winY[-nPixY/2:]		

	return np.outer(winOutX, winOutY)

class lowRankDeconv(object):
	"""
	Class that carries out a deconvolution assuming that the Stokes profiles are low-rank	
	"""
	def __init__(self, stokes, psf):
		"""
		Creator of the class
		
		Args:
		    stokes (float): Stokes parameters cube [nx, ny, nlambda]. The data should be periodic or be apodized
		    to avoid artifacts
				
		"""			
		self.nx, self.ny, self.nl = stokes.shape
		self.stokes = stokes

# Compute the original PCA decomposition
		M = self.stokes.reshape((self.nx*self.ny,self.nl))
		corrM = M.T.dot(M)
		uSVD, wSVD, vSVD = np.linalg.svd(corrM)
		self.baseOrig = uSVD				

		self.psf = psf		

		self.psfFFT = np.fft.fft2(self.psf)
			
		self.mu = 1.0	

	def forward(self, z):
		return np.fft.ifft2(self.psfFFT * np.fft.fft2(z))

	def backward(self, z):
		return np.fft.ifft2(np.conj(self.psfFFT) * np.fft.fft2(z))

	def FISTA(self, rank=5, niter=10, padX=0, padY=0, wavelength=0):
		"""
		Carry out the deconvolution using the FISTA algorithm, that solves the following problem:

		argmin_O  ||I - P*O||, s.t. rank(O) = r

		where I is the observed Stokes parameters on all pixels and P is the PSF
		The rank is obtained by ordering all pixels in lexicographic order
		
		Args:
		    rank (int, optional): rank of the solution
		    niter (int, optional): number of iterations
		
		Returns:
		    TYPE: Description
		"""
		self.x = np.copy(self.stokes)
		self.xNew = np.copy(self.stokes)
		self.y = np.copy(self.stokes)

		t = 1.0
		self.std = []
		for loop in range(niter):			
			for iLambda in range(self.nl):
				z = np.copy(self.y[:,:,iLambda])
				forw = self.forward(z)
				residual = self.backward(forw - self.stokes[:,:,iLambda])
				self.xNew[:,:,iLambda] = self.y[:,:,iLambda] - self.mu * np.real(residual)

			M = self.xNew.reshape((self.nx*self.ny,self.nl))
			corrM = M.T.dot(M)
			uSVD, wSVD, vSVD = np.linalg.svd(corrM)

			self.base = uSVD
			coefs = M.dot(self.base[:,0:rank])
			output = coefs.dot(self.base[:,0:rank].T)

			self.xNew = output.reshape((self.nx,self.ny,self.nl))

			tNew = 0.5*(1+np.sqrt(1+4.0*t**2))

			self.y = self.xNew + (t-1.0) / tNew * (self.xNew - self.x)

			t = tNew

			self.x = np.copy(self.xNew)
			std = np.std(self.x[padX:-padX,padY:-padY,wavelength])
			self.std.append(std)
			print "Iteration {0} - std={1} - constrast={2}%".format(loop, std, np.std(self.x[padX:-padX,padY:-padY,wavelength])/np.mean(self.x[padX:-padX,padY:-padY,wavelength])*100.0)
			

		self.std = np.asarray(self.std)
		self.stIDeconv = self.x

		return self.stIDeconv
	

# Number of PCA coefficients to keep in each Stokes parameter
nPCA = [6,4,4,4]
labels = ['sti', 'stq', 'stu', 'stv']
nIter = [15,10,10,10]

deconv = [None] * 4

for loopStokes in range(4):
	label = labels[loopStokes]

# stokes is a variable with size [nx,ny,nlambda] for each Stokes parameter
	# stokes = 
	nx, ny, nl = stokes.shape
	nPixBorder = nx / 2

# Choose among a reflecting padding or an apodization to make the map periodic
	# Pad tha map to make it somehow
	# stokes = np.pad(stokes, ((nPixBorder,nPixBorder), (0,0), (0,0)), mode='reflect')	

# In this case, I choose an apodization mask with 12 pixels
	# Apodization mask
	window = hanningWindow(nx,ny,12,12)
	stokes *= window[:,:,None]

	# Read the PSF and make it the same size as the data. The PSF has size 65x65
	# Change the way to read the PSF from your HD
	psfOrig = np.load('psf_0.5arcsec.npy')
	psfOrig = np.roll(np.roll(psfOrig,32,axis=0),32,axis=1)
	psfOrig = psfOrig[2:-3,2:-3]
	psfSize = psfOrig.shape[0]
			
# Put is in the appropriate place in the map
	psf = np.zeros(stokes.shape[0:2])
	psf[:,ny/2-psfSize/2:ny/2+psfSize/2] = psfOrig
	psf = np.roll(psf, nx/2, axis=0)
	psf = np.roll(psf, ny/2, axis=1)

	# Normalize to unit area
	psf /= np.sum(psf)

	# Instantiate the class
	out = lowRankDeconv(stokes, psf)

	# Carry out the deconvolution using 20 iterations and asuming rank-5 for the data
	out.FISTA(rank=nPCA[loopStokes], niter=nIter[loopStokes], padX=nPixBorder, padY=12, wavelength=201)

# Cut the piece without apodization
	orig, deconv = out.stokes, out.stIDeconv
	orig = orig[12:-12,12:-12,:]
	deconv = deconv[12:-12,12:-12,:]

	stokesDeconv[loopStokes] = deconv

stokesDeconv = np.asarray(stokesDeconv)