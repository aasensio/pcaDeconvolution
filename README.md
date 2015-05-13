# pcaDeconvolution

Deconvolution of maps of Stokes parameters using a PCA regularization

This deconvolution method follows the scheme presented in Ruiz Cobo & Asensio Ramos (2013)
The Stokes parameters are projected onto a few spectral eigenvectors and the ensuing maps
of coefficients are deconvolved using a standard Lucy-Richardson algorithm. This introduces
a stabilization because the PCA filtering reduces the amount of noise.

In order to use it, make sure you read the Stokes parameters and run the `deconvolveAll`
routine.

	stokes = ['I','Q','U','V']
	npca = [10,10,10,10]
	iter = [50,25,25,25]
	for i = 0, 3 do begin

	; Read your Stokes parameter file and enter it into the next routine
	; For instance, use restore, 'stokes'+stokes+'.idl' if you have the
	; Stokes parameters saved on different files
		precalculatePCA, data, stokes[i]
		computePCAMaps, stokes[i], npca[i]
		deconvolutionPCA, stokes[i], iter[i], /filter, npca=npca[i]
	endfor
