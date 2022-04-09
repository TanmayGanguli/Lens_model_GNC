# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:02:09 2022

@author: Tanmay Ganguli
"""
import numpy as np
import matplotlib.pyplot as plt
import math,cmath
from scipy.fftpack import fft,ifft,fft2,ifft2,fftshift,ifftshift

def Le_polar(m_x,m_y):
    #returns polar coordinates given cartesian ones
    r = np.sqrt(m_x*m_x+m_y*m_y)
    theta = np.arctan2(m_y,m_x)
    return r,theta


def Le_delta_phase(coefficients_st, m_r, m_t):
    #calculate change of phase at given point (r,theta)
    #use zernike polynomials with given coeff
    #input is in form of an array of rad,theta for all points under consideration
    a = coefficients_st
    Z1  =  a[0]  * 1*(np.cos(m_t)**2+np.sin(m_t)**2)
    Z2  =  a[1]  * 2*m_r*np.cos(m_t)
    Z3  =  a[2]  * 2*m_r*np.sin(m_t)
    Z4  =  a[3]  * np.sqrt(3)*(2*m_r**2-1)
    Z5  =  a[4]  * np.sqrt(6)*m_r**2*np.sin(2*m_t)
    Z6  =  a[5]  * np.sqrt(6)*m_r**2*np.cos(2*m_t)
    Z7  =  a[6]  * np.sqrt(8)*(3*m_r**2-2)*m_r*np.sin(m_t)
    Z8  =  a[7]  * np.sqrt(8)*(3*m_r**2-2)*m_r*np.cos(m_t)
    Z9  =  a[8]  * np.sqrt(8)*m_r**3*np.sin(3*m_t)
    Z10 =  a[9] * np.sqrt(8)*m_r**3*np.cos(3*m_t)
    Z11 =  a[10] * np.sqrt(5)*(1-6*m_r**2+6*m_r**4)
    Z12 =  a[11] * np.sqrt(10)*(4*m_r**2-3)*m_r**2*np.cos(2*m_t)
    Z13 =  a[12] * np.sqrt(10)*(4*m_r**2-3)*m_r**2*np.sin(2*m_t)
    Z14 =  a[13] * np.sqrt(10)*m_r**4*np.cos(4*m_t)
    Z15 =  a[14] * np.sqrt(10)*m_r**4*np.sin(4*m_t)
    Z16 =  a[15] * np.sqrt(12)*(10*m_r**4-12*m_r**2+3)*m_r*np.cos(m_t)
    Z17 =  a[16] * np.sqrt(12)*(10*m_r**4-12*m_r**2+3)*m_r*np.sin(m_t)
    Z18 =  a[17] * np.sqrt(12)*(5*m_r**2-4)*m_r**3*np.cos(3*m_t)
    Z19 =  a[18] * np.sqrt(12)*(5*m_r**2-4)*m_r**3*np.sin(3*m_t)
    Z20 =  a[19] * np.sqrt(12)*m_r**5*np.cos(5*m_t)
    Z21 =  a[20] * np.sqrt(12)*m_r**5*np.sin(5*m_t)
    Z22 =  a[21] * np.sqrt(7)*(20*m_r**6-30*m_r**4+12*m_r**2-1)
    Z23 =  a[22] * np.sqrt(14)*(15*m_r**4-20*m_r**2+6)*m_r**2*np.sin(2*m_t)
    Z24 =  a[23] * np.sqrt(14)*(15*m_r**4-20*m_r**2+6)*m_r**2*np.cos(2*m_t)
    Z25 =  a[24] * np.sqrt(14)*(6*m_r**2-5)*m_r**4*np.sin(4*m_t)
    Z26 =  a[25] * np.sqrt(14)*(6*m_r**2-5)*m_r**4*np.cos(4*m_t)
    Z27 =  a[26] * np.sqrt(14)*m_r**6*np.sin(6*m_t)
    Z28 =  a[27] * np.sqrt(14)*m_r**6*np.cos(6*m_t)
    Z29 =  a[28] * 4*(35*m_r**6-60*m_r**4+30*m_r**2-4)*m_r*np.sin(m_t)
    Z30 =  a[29] * 4*(35*m_r**6-60*m_r**4+30*m_r**2-4)*m_r*np.cos(m_t)
    Z31 =  a[30] * 4*(21*m_r**4-30*m_r**2+10)*m_r**3*np.sin(3*m_t)
    Z32 =  a[31] * 4*(21*m_r**4-30*m_r**2+10)*m_r**3*np.cos(3*m_t)
    Z33 =  a[32] * 4*(7*m_r**2-6)*m_r**5*np.sin(5*m_t)
    Z34 =  a[33] * 4*(7*m_r**2-6)*m_r**5*np.cos(5*m_t)
    Z35 =  a[34] * 4*m_r**7*np.sin(7*m_t)
    Z36 =  a[35] * 4*m_r**7*np.cos(7*m_t)
    Z37 =  a[36] * 3*(70*m_r**8-140*m_r**6+90*m_r**4-20*m_r**2+1)
    
    ans = Z1 + Z2 +  Z3+  Z4+  Z5+  Z6+  Z7+  Z8+  Z9+ Z10+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
    return ans

#define function which returns exit pupil radius
def Le_exit_pupil_radius(aperture, f, wavelength, pixel, side):
    theta_pixel = side*pixel/f
    theta_screen = wavelength/aperture
    radius = int(theta_pixel/2/theta_screen)
    return radius



#define pupil function
def Le_pupil_function(l, b, rpupil, r_coefficients_st):
    r = 1
    m_pupil_matrix = np.zeros([l, b], dtype=complex)
    #final array containing value of pupilfunc at each point
    #mask function
    r_x = np.linspace(-r, r, 2*rpupil)
    r_y = np.linspace(-r, r, 2*rpupil)
    [m_X, m_Y] = np.meshgrid(r_x, r_y)#store values of x,y coordinates in X,Y
    m_rad,m_theta = Le_polar(m_X, m_Y)
    m_M = 1*(np.sin(m_theta)**2 + np.cos(m_theta)**2)
    
    m_M[m_rad > 1] = 0
    #phase modulation
    m_A = np.exp(1*cmath.sqrt(-1)*Le_delta_phase(r_coefficients_st, m_rad, m_theta), dtype=complex)
    m_pupil_center = m_M*m_A
    m_pupil_matrix[l//2-rpupil+1:l//2+rpupil+1,b//2-rpupil+1:b//2+rpupil+1] = m_pupil_center
    return m_pupil_matrix

#define PSF
def Le_PSF(m_pupil):
    #fourier transform of pupil function
    m_psf = fftshift(fft2(ifftshift(m_pupil)))
    m_psf = np.abs(m_psf)**2
    return m_psf/m_psf.sum()
    
#define OTF
def Le_OTF(m_psf):
    #fourier transform of psf
    m_otf = fftshift(fft2(ifftshift(m_psf)))
    m_otf = m_otf/abs(m_otf.max())
    return m_otf


#define MTF
def Le_MTF(m_otf):
    return np.abs(m_otf)



#__main__


#given variables in metres
APERTURE = 0.08
FOCUS = 0.8
WAVELENGTH = 550*10**(-9)
PIXEL_SIZE = 5*10**(-6)
LENGTH = 1000 #sensor length
BREADTH = 1200 #sensor width
r_coefficients_st = np.zeros(37)#zernike coefficients for phase modulation



r_coefficients_st[0] = 6.54080697
r_coefficients_st[3] = 3.77411473
r_coefficients_st[10] = -0.00172336
r_coefficients_st[21] = -0.00000190




#calculations
pupil_radius = Le_exit_pupil_radius(APERTURE,FOCUS,WAVELENGTH,PIXEL_SIZE,(LENGTH+BREADTH)/2)
m_pupil = Le_pupil_function(LENGTH,BREADTH,pupil_radius,r_coefficients_st)
m_psf = Le_PSF(m_pupil)
m_otf = Le_OTF(m_psf)
m_mtf = Le_MTF(m_otf)


#plot figures
#plot psf 
plt.imshow(m_psf)
plt.xlim(500,700)
plt.ylim(400,600)
plt.show()

plt.imshow(m_psf)
plt.xlim(0,1200)
plt.ylim(0,1000)
plt.show()
'''
plt.plot(m_psf[499])
plt.xlim(500,700)
plt.show()
'''
#plot mtf

plt.imshow(m_mtf)
plt.xlim(0,1200)
plt.ylim(0,1000)
plt.show()
'''
plt.plot(m_mtf[499])
plt.xlim(0,1200)
plt.show()
'''

    

    
    
    
    
    
    
    
    
    
    