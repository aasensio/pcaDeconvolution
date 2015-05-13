; Performs the PCA decomposition of a dataset, returning the eigenvalues and the
; eigenvectors
; INPUT
;  A: NxM matrix that contains, as rows, the N M-dimensional observations
;    If the dataset are images of dimensions PxQ, we put each row as a M-dimensional vector
;    of size P*Q
; OUTPUT
;  eigenvalues: the set of M eigenvalues
;  eigenvectors: a matrix of size MxM containing the eigenvectors
;        eigenvectors[0,*] is the first eigenvector, associated with eigenvalues[0]
;        eigenvectors[i,*] is the i-th eigenvector, associated with eigenvalues[i]
;  
pro mypca, A, eigenvalues, eigenvectors, regularization=regularization
	print, 'Calculating correlation matrix...'
	s = size(A)
	if (s[1] gt s[2] and s[1] gt 1000) then begin
		print, 'You should consider transposing the data
		return
	endif
	corr = transpose(A)##A
	print, 'SVD decomposition...'
	if (keyword_set(regularization)) then begin
		corr = corr + 1.d-8*max(corr)
	endif
	svdc, corr, w, u, v
	
	eigenvectors = v
	eigenvalues = w
end
