; Read the observations and compute the PCA eigenvectors
; stokes is of size [nlambda,nx,ny]
; stokesParameter is 'I', 'Q', 'U' or 'V'. This will be appended to the name of the output file
pro precalculatePCA, stokes, stokesParameter
	
	nl = n_elements(stokes[*,0,0])
	nx = n_elements(stokes[0,*,0])
	ny = n_elements(stokes[0,0,*])
		
	h = reform(stokes,nl,1L*nx*ny)
	mn = fltarr(nl)
	for i = 0, nl-1 do mn[i] = mean(h[i,*])
 	
; Substract the mean
	h = h - mn#replicate(1.0,1L*nx*ny)
 	
; And do the PCA decomposition
	mypca, h, eval, evec
		
	save, stokes, filename='stokes'+stokesParameter+'_orig.idl'
	save, eval, evec, mn, filename='stokes'+stokesParameter+'_eigenvec.idl'	
end

; Compute the PCA maps by projecting the data on the PCA eigenvectors
pro computePCAMaps, stokesParameter, nPCACoefficients

	root = 'stokes'+stokesParameter
	print, 'Reading Stokes '+stokesParameter
	restore,root+'_orig.idl'
	restore,root+'_eigenvec.idl'
		
	print, 'Projecting on the '+strtrim(string(nPCACoefficients),2)+' PCA eigenvectors...'
	s = size(stokes)
	nlambda = s[1]
	nf = s[3]
	nslit = s[2]
		
	h = reform(stokes,nlambda,1L*nf*nslit)
	h = h - mn#replicate(1.0,1L*nf*nslit)

	base = evec[0:nPCACoefficients-1,*]

	coef = h ## base
	h = 0.0
	coef = reform(coef, nPCACoefficients, nslit, nf)

	print, 'Saving maps of PCA coefficients...'
	save, coef, filename=root+'_PCAMaps.idl'
	coef = 0.0
	stokes = 0.0
	
end

; PCA-regularized deconvolution
; Deconvolve the Stokes parameters using a Richardson-Lucy algorithm
; stokesParameter: 'I', 'Q', 'U', 'V'
; psfFile : FITS file containing the PSF
; maxIter : maximum number of iterations
; npca : number of PCA eigenvectors to consider
; apodization : percentage of the image to apodize
;
; Example
; deconvolutionPCA, 'I', 50, npca=4
pro deconvolutionPCA, stokesParameter, psfFile, maxIter, npca=npca, apodization=apodization

	if (not keyword_set(apodization)) then apodization = 1.d0 / 20.d0
		
; Read the original data
	print, 'Reading PCA coefficients and eigenvectors...'
	root = 'stokes'+stokesParameter
	restore,root+'_PCAMaps.idl'
	restore,root+'_eigenvec.idl'
		
	s = size(coef)
	dimx = s[2]
	dimy = s[3]
	nlam = n_elements(eval)
		
	restore,root+'_orig.idl'
		
	noise_lev = 1.1d-3
				
; Compute a mask to avoid all points without signal above 3 times the noise level
	if (keyword_set(filter) and stokesParameter ne 'I') then begin
		maxim = max(abs(stokes),dimension=1)
		mask = replicate(1,dimx,dimy)
		ind = where(maxim lt 3.d0 * noise_lev)
		mask[ind] = 0
	endif else begin
		mask = replicate(1,dimx,dimy)
	endelse

; We won't need the original data anymore
	stokes = 0.0

; Read the PSF. We assume the PSF is given in a square array. Note that it should be odd
	psf = readfits(psfFile)
	psfSize = n_elements(psf[*,0])

; Reconstruct a large map with the PSF in the center
	psf_large = fltarr(dimx,dimy)
	psf_large[dimx/2-psfSize/2:dimx/2+psfSize/2,dimy/2-psfSize/2:dimy/2+psfSize/2] = psf

; Renormalize the total area to unity
	psf_large = psf_large / total(psf_large)

; Apply a window to the image to avoid high frequencies in the border
	win = red_taper2(dimx,dimy,apodization)

; Make a copy of the coefficients to compute the reconstructed coefficients
	coef_recons = coef
	
	!p.multi = [0,2,1]
	window,0,xsize=512*2L+1,ysize=513

	error = fltarr(maxIter+1)
	for i = 0, npca-1 do begin
	
		wset,0

; Take the i-th PCA coefficient, apply the window
; and normalize with the median
		dat = reform(coef[i,*,*]) * mask
		
		mdat = mean(dat)
		dat = dat / mdat * win

		deconv_old = dat
		deconv_orig = dat
		
		error[0] = 1.d0
		j = 0
; Carry out the deconvolution for a maximum of maxIter iterations
		while (error[j] gt 1.d-4 and j lt maxIter) do begin			
	
			j = j + 1
			max_likelihood, dat, psf_large, deconv, /gaussian, ft_psf=psf_ft
			
			res = convolve(deconv, psf_large)

			error[j] = total((res-deconv_orig)^2) / (1.d0*dimx*dimy)
			print, 'Iter: ', j, ' - Error : ', error[j], ' - NPCA : ', i
			if (j ne 0) then begin
				if (error[j] gt error[j-1]) then begin
					deconv = deconv_old
					print, 'Error is increasing. Going back to the previous image...'
				endif else begin
					deconv_old = deconv
				endelse
			endif
			
			tvframe,dat*mdat
			tvframe,deconv*mdat
			
		endwhile
		
		coef_recons[i,*,*] = deconv * mdat

	endfor

	save, filename= root+'_deconvolvedPCAMaps.idl', coef_recons
		
	!p.multi = 0

end

; Reconstruct the deconvolved maps
; stokesParameter : 'I', 'Q', 'U', 'V'
; stokesOrigPCA : original profiles projected on the subspace of npca eigenvectors
; stokesDeconvolvedPCA : deconvolved profiles projected on the subspace of npca eigenvectors
; npca : number of PCA eigenvectors to use on the reconstruction
pro reconstructDeconvolved, stokesParameter, stokesOrigPCA, stokesDeconvolvedPCA, npca=npca
	
	root = 'stokes'+stokesParameter
	restore,root+'_eigenvec.idl'
	restore,root+'_deconvolvedPCAMaps.idl'
	restore,root+'_PCAMaps.idl'
	
	nx = 1L*n_elements(coef_recons[0,*,0])
	ny = 1L*n_elements(coef_recons[0,0,*])
	nlambda = n_elements(evec[*,0])
		
	if (not keyword_set(npca)) then begin
		npca = n_elements(coef_recons[*,0,0])
	endif
		
	stokesDeconvolvedPCA = reform(coef_recons[0:npca-1,*,*],npca,nx*ny) ## transpose(evec[0:npca-1,*]) + mn # replicate(1.d0, nx*ny)
	stokesDeconvolvedPCA = reform(stokesDeconvolvedPCA, nlambda, nx, ny)
	
	stokesOrigPCA = reform(coef[0:npca-1,*,*],npca,nx*ny) ## transpose(evec[0:npca-1,*]) + mn # replicate(1.d0, nx*ny)
	stokesOrigPCA = reform(stokesOrigPCA, nlambda, nx, ny)
end

; Deconvolve all Stokes parameters using a different number of iterations and PCA eigenvectors
; for each one
pro deconvolveAll
	stokes = ['I','Q','U','V']
	npca = [10,10,10,10]
	iter = [50,25,25,25]
	for i = 0, 3 do begin

; Read your Stokes parameter file and enter it into the next routine
; For instance, use restore, 'stokes'+stokes+'.idl' if you have the
; Stokes parameters saved on different files
		precalculatePCA, data, stokes[i]
		computePCAMaps, stokes[i], npca[i]
		deconvolutionPCA, stokes[i], iter[i], npca=npca[i]
	endfor
end
