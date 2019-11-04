# histogramEqualisation-spatialDerivative-backProjection

a. Read a RGB colour image (e.g. png) and display it.
b. Convert the RGB image into Lab colour system and display each component (L,a,b) as an
grey level image (See section 2.7.4 in
http://www.ee.columbia.edu/ln/dvmm/publications/PhD_theses/jrsmith-thesis.pdf).
c. Compute the spatial derivatives of the luminance component L in the horizontal and vertical direction using convolution by the derivatives of Gaussian filter. Display each
these derivatives as grey level images.
d. Compute a 2D histogram with the chrominance component (a,b) and display the
histogram as a grey image (heat map) and/or as a 3D surface (bar plot)
e. Using a part of the image to compute a 2D histogram model with the chrominance component (a,b), compute a back projection map with this model histogram in the target
image (i.e. see BP1 section 5.3 in
http://www.ee.columbia.edu/dvmm/publications/PhD_theses/jrsmith-thesis.pdf ).
f. Perform histogram equalization using 1D histogram using the luminance L computed in 1.a. Display the resulting image with enhanced contrast (e.g.
https://en.wikipedia.org/wiki/Histogram_equalization )
g. Evaluate the performance of these techniques with histograms (i.e. illustrate when it
works, and when it does not work).
