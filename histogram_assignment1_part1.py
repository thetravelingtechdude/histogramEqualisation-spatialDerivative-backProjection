# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from mpl_toolkits.mplot3d import Axes3D






def twoDHistogram(x, y, img):
    #Source - https://matplotlib.org/3.1.1/gallery/mplot3d/hist3d.html
    #Takes the LAB image, its a* and b* component as the input and returns a 2D histogram of the image
    fig = plt.figure()
    plt.title('3D bar plot of 2D histogram of the LAB image')
    ax = fig.add_subplot(111, projection='3d')
    
    #x, y = np.random.rand(2, 100) * 4
    #Compute the 2D histogram
    hist, xedges, yedges = np.histogram2d(x.ravel(), y.ravel(), bins=100)

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort= 'average')

    return hist
 




def computeSpatialDerivative(x):
    #Source - https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    #Source - https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.diff.html
    
    #Apply Gaussian filter to the luminance component
    filteredImage = ndi.gaussian_filter(x,sigma = 5)
    plt.figure()
    plt.title('After gaussian filter')
    plt.imshow(filteredImage,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Computing the spatial derivative of the luminance component in the horixontal (x) direction
    lx_derivative = np.diff(filteredImage, axis = 0)  #axis = 0 indicates X axis
    plt.figure()
    plt.title('Spatial derivative of the luminance component in the horixontal (x) direction')
    plt.imshow(lx_derivative,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Computing the spatial derivative of the luminance component in the vertical (y) direction
    ly_derivative = np.diff(filteredImage, axis = 1)  #axis = 1 indicates Y axis
    plt.figure()
    plt.title('Spatial derivative of the luminance component in the vertical (y) direction')
    plt.imshow(ly_derivative,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return x





def twoDBackProjection(l_original,a_original,b_original,lab_original_img):
    #Source - https://pysource.com/2018/03/30/histogram-and-back-projection-opencv-3-4-with-python-3-tutorial-28/
    #Source - http://www.ee.columbia.edu/ln/dvmm/publications/PhD_theses/jrsmith-thesis.pdf
    
    #Read the cropped image
    cropped_img = plt.imread('cropped_img1.jpg')
    
    #Display the cropped image
    plt.figure()
    plt.title('Cropped image')
    plt.imshow(cropped_img,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Convert the cropped image to LAB format
    lab_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
    
    #Display LAB cropped image
    #Display the cropped image
    plt.figure()
    plt.title('LAB Cropped image')
    plt.imshow(lab_cropped_img,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Splitting L, a* and b* components of the cropped image
    l_cropped, a_cropped, b_cropped = cv2.split(lab_cropped_img)
    
    # Histogram of the cropped image
    lab_cropped_hist = cv2.calcHist([lab_cropped_img],[1,2],None,[256,256],[0,256,0,256])

    #lab_cropped_hist = twoDHistogram(a_cropped, b_cropped, lab_cropped_img)
    mask = cv2.calcBackProject([lab_original_img], [1, 2], lab_cropped_hist, [0, 256, 0, 256], 1)
    
    #Generating an elliptical kernel of array size 5x5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    #Convolution of the mask with the kernel
    mask = cv2.filter2D(mask, -1, kernel)
    
    #Perform threasholding
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    
    #Merging the 3 components of the mask 
    mask = cv2.merge((mask, mask, mask))
    
    #Bitwise AND operation of the original LAB image and the mask 
    result = cv2.bitwise_and(lab_original_img, mask)
    
    '''
    #Display the original image
    plt.figure()
    plt.title('Original LAB image')
    plt.imshow(lab_original_img,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    #Display the mask
    plt.figure()
    plt.title('Mask')
    plt.imshow(mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    
    #Display the result of back projection
    plt.figure()
    plt.title('Result')
    plt.imshow(result,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return




def histEqualization(l_comp):
    #Source: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
    #Source: Pg. 24, Programming Computer Vision with Python, Jan Erik Solem
    #http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf
    
    #Compute 1D histogram of the luminance component
    hist_l_comp,bins = np.histogram(l_comp.flatten(),256)
    
    #The transform function is in this case a cumulative distribution
    #function (cdf) of the pixel values in the image (normalized to map
    #the range of pixel values to the desired range).
    
    cdf = hist_l_comp.cumsum() #cumulative distribution function 
    cdf = 255 * cdf / cdf[-1] #normalize to map the range of pixel values to the desired range
    
    # use linear interpolation of cdf to find new pixel values
    im_hist_eq = np.interp(l_comp.flatten(),bins[:-1],cdf) 
    hist_eq = im_hist_eq.reshape(l_comp.shape)
    
    plt.figure()
    plt.title('After Histogram equalization of the Luminance component')
    plt.imshow(hist_eq,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return 


  
 
    
def main():
    #Load a color image and show it
    img = plt.imread('img1.jpg')
    #print(img)
    
    #Display the image
    plt.figure()
    plt.title('Image')
    plt.imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Converting the image from RGB to LAB
    LABimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_comp, a_comp, b_comp = cv2.split(LABimg) 
    
    #Display the L component
    plt.figure()
    plt.title('L')
    plt.imshow(l_comp,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Display the a* component
    plt.figure()
    plt.title('a*')
    plt.imshow(a_comp,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Display the b* component
    plt.figure()
    plt.title('b*')
    plt.imshow(b_comp,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Computing spatial derivative of the luminance component
    computeSpatialDerivative(l_comp)

    #Computing a 2D histogram with the chrominance (a,b) component
    twoDHistogram(a_comp, b_comp, LABimg)
    
    #Performing histogram equilization for a 1D histogram using Luminance component
    histEqualization(l_comp)
    
    #Computing back projection map using 2D Histogram with the chrominance (a,b) component
    twoDBackProjection(l_comp,a_comp,b_comp,LABimg)

if __name__ == '__main__':
    main()
