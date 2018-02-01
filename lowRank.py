from __future__ import print_function
import numpy as np
import scipy.io
import scipy.signal
from astropy.io import fits
from ipdb import set_trace as stop

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
	winOutX[0:int(nPixX/2)] = winX[0:int(nPixX/2)]
	winOutX[-int(nPixX/2):] = winX[-int(nPixX/2):]

	winOutY = np.ones(ny)
	winOutY[0:int(nPixY/2)] = winY[0:int(nPixY/2)]
	winOutY[-int(nPixY/2):] = winY[-int(nPixY/2):]		

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
			std = np.std(self.x[padX:-padX,padY:-padY,wavelength-1])
			self.std.append(std)
			print("Iteration {0} - std={1} - constrast={2}%".format(loop, std, np.std(self.x[padX:-padX,padY:-padY,wavelength-1])/np.mean(self.x[padX:-padX,padY:-padY,wavelength-1])*100.0))
			

		self.std = np.asarray(self.std)
		self.stIDeconv = self.x

		return self.stIDeconv
	
if (__name__ == '__main__'):
	# Number of PCA coefficients to keep in each Stokes parameter
	nPCA = [6,4,4,4]
	labels = ['sti', 'stq', 'stu', 'stv']
	nIter = [5,10,10,10]

	stokes_deconv = [None] * 4

	for loopStokes in range(1):
		label = labels[loopStokes]

	# stokes is a variable with size [nx,ny,nlambda] for each Stokes parameter
		tmp = scipy.io.readsav('smallPatch.idl')
		stokes = tmp['data']
		nx, ny, nl = stokes.shape
		nPixBorder = int(nx / 2)

	# Choose among a reflecting padding or an apodization to make the map periodic
		# Pad tha map to make it somehow
		# stokes = np.pad(stokes, ((nPixBorder,nPixBorder), (0,0), (0,0)), mode='reflect')	

	# In this case, I choose an apodization mask with 12 pixels
		# Apodization mask
		window = hanningWindow(nx,ny,12,12)
		stokes *= window[:,:,None]

		# Read the PSF and make it the same size as the data. The PSF has size 65x65
		# Change the way to read the PSF from your HD
		f = fits.open('hinode_psf.0.16-df.fits')
		psfOrig = f[0].data
		psfOrig = psfOrig[2:-2,2:-2]
		psfSize = psfOrig.shape[0]

		psf = np.zeros((nx,ny))
		psf[int(nx/2-psfSize/2+1):int(nx/2+psfSize/2+1),int(ny/2-psfSize/2+1):int(ny/2+psfSize/2+1)] = psfOrig
		psf = np.fft.fftshift(psf)
					
		# Normalize to unit area
		psf /= np.sum(psf)

		# Instantiate the class
		out = lowRankDeconv(stokes, psf)

		# Carry out the deconvolution using 20 iterations and asuming rank-5 for the data
		out.FISTA(rank=nPCA[loopStokes], niter=nIter[loopStokes], padX=12, padY=12, wavelength=112)

	# Cut the piece without apodization
		orig, deconv = out.stokes, out.stIDeconv
		orig = orig[12:-12,12:-12,:]
		deconv = deconv[12:-12,12:-12,:]

		stokes_deconv[loopStokes] = deconv

	stokes_deconv = np.asarray(stokes_deconv)