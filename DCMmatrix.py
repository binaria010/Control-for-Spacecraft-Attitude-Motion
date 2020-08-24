#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:02:44 2020

@author: Juliana
"""

import numpy as np
import math
import scipy
from scipy import optimize
from scipy.linalg import *

def M_axis(theta,axis):
    """
    This function computes a rotation matrix around a given axis

    Parameters
    ----------
    theta : double given in degrees. Angle of rotation
    axis : int. 1, 2 or 3. 1 = x axis, 2 = y axis, 3 = z axis

    Returns
    -------
    M : 3x3 ndarray. the matrix of rotation
    """
    
    theta = theta*np.pi /180 # converts from degs to radians
    M = np.zeros((3,3))
    
    if axis == 1:
        M[0,:] = np.array([1, 0, 0])
        M[1,:] = np.array([0, np.cos(theta), np.sin(theta)])
        M[2,:] = np.array([0, -np.sin(theta), np.cos(theta)])
    
    if axis == 2:
        M[0,:] = np.array([np.cos(theta), 0, -np.sin(theta)])
        M[1,:] = np.array([0, 1, 0])
        M[2,:] = np.array([np.sin(theta), 0, np.cos(theta)])

    if axis == 3:
        M[0,:] = np.array([np.cos(theta), np.sin(theta), 0])
        M[1,:] = np.array([-np.sin(theta), np.cos(theta), 0])
        M[2,:] = np.array([0, 0, 1])

    return M

def DCMatrix(Theta,euler):
    """
    This function computes de direcion cosinde matrix (DCM) of an Euler angle
    set with angles (Theta = theta1, thet2, theta3)

    Parameters
    ----------
    Theta : tuple of thetas
    euler : tuple of integers giving the axis around to rotate the angles.

    Theta[0] = theta1 : float. angle around the first axis in euler: in degs
    Theta[1] = theta2 : float. angle around the second axis in euler: in degs
    Theta[2] = theta3 : float. angle around the third axis in euler: in degs
    

    Returns
    -------
    The DCM marix. If Theta = (t1,t2,t3) and euler = (a,b,c) the the DCM matrix
    is given by the multiplication of he three matrices:
        C = M(t3, c) @ M(t2, b) @ M(t1, a)

    """

    
    C = M_axis(Theta[2],euler[2])@ M_axis(Theta[1],euler[1]) @ M_axis(Theta[0],euler[0])
    
    return C

def rotation_angles(C, euler) :
    """
    This function returns the Euler angle set of a rotation of type euler

    Parameters
    ----------
    C : 3x3 ndarray. Direction cosine matrix DCM
    euler : tuple of integers giving the type of rotation

    Returns
    -------
    tuple of floats: (theta1, theta2, theta3) in degrees

    """
    if euler == (1,2,1) :
        theta1 = math.atan2(C[0,1],-C[0,2])
        theta2 = math.acos(C[0,0])
        theta3 = math.atan2(C[1,0], C[2,0])

    if euler == (1,2,3) :
        theta1 = math.atan2(-C[2,1],C[2,2])
        theta2 = math.asin(C[2,0])
        theta3 = math.atan2(-C[1,0], C[0,0])
        
    if euler == (1,3,1) :
        theta1 = math.atan2(C[0,1],C[0,2])
        theta2 = math.acos(C[0,0])
        theta3 = math.atan2(C[2,0], -C[1,0])

    if euler == (1,3,2) :
        theta1 = math.atan2(C[1,2],C[1,1])
        theta2 = - math.sin(C[1,0])
        theta3 = math.atan2(C[2,0], C[0,0])
    
    if euler == (2,1,2) :
        theta1 = math.atan2(C[1,0], C[1,2])
        theta2 = math.acos(C[1,1])
        theta3 = math.atan2(C[0,1], -C[2,1])
    
    if euler == (2,1,3) :
        theta1 = math.atan2(C[2,0], C[2,2])
        theta2 = -math.asin(C[2,1])
        theta3 = math.atan2(C[0,1], C[1,1])

    if euler == (2,3,1) :
        theta1 = math.atan2(-C[0,2], C[0,0])
        theta2 = math.asin(C[0,1])
        theta3 = math.atan2(-C[2,1], C[1,1])
        
    if euler == (2,3,2) :
        theta1 = math.atan2(C[1,2], -C[1,0])
        theta2 = math.acos(C[1,1])
        theta3 = math.atan2(C[2,1], C[0,1])
        
    if euler == (3,1,2) :
        theta1 = math.atan2(-C[1,0], C[1,1])
        theta2 = math.asin(C[1,2])
        theta3 = math.atan2(-C[0,2], C[2,2])
        
    if euler == (3,1,3) :
        theta1 = math.atan2(C[2,0], -C[2,1])
        theta2 = math.acos(C[2,2])
        theta3 = math.atan2(C[0,2], C[1,2])
        
    if euler == (3,2,1) :
        theta1 = math.atan2(C[0,1], C[0,0])
        theta2 = -math.asin(C[0,2])
        theta3 = math.atan2(C[1,2], C[2,2])
        
    if euler == (3,2,3) :
        theta1 = math.atan2(C[2,1], C[2,0])
        theta2 = math.acos(C[2,2])
        theta3 = math.atan2(C[1,2], -C[0,2])

    return theta1*180/np.pi, theta2*180/np.pi, theta3*180/np.pi


def Matrix_tilde(w):
    """
    This function computes the matrix of the linear transformation given by:
        w x _ : R^3\to R^3 the croos product on the left with w

    Parameters
    ----------
    w : 1 D numpy array.
    Returns
    -------
    W : 3x3 ndarray. A skew symmetric matrix

    """
    w1, w2, w3 = w
    
    W = np.zeros((3,3))
    W[0,1] = - w3
    W[0,2] = w2
    W[1,0] = w3
    W[1,2] = -w1
    W[2,0] = -w2
    W[2,1] = w1
    
    return W



def PR_to_DCM(Phi, e):
    """
    This function returns the DCM mtrix or basis change matrix from frame N to 
    frame B with principal rotation angle Phi around the principal rotation vector e.

    Parameters
    ----------
    Phi : Double. angle of rotation in degrees. Principal Rotation Angle
    e : 1d numpy array column vector (important to be a 2 dimensional array so e@e.T computes the exterior product!. Represents the Principal Rotation Vector  

    Returns
    -------
    C : 3x3 numpy array. The DCM matrix from frame N to frame B. Usually denoted [BN]

    """
    Phi = Phi* np.pi/180
    
    C = math.cos(Phi)* np.eye(3) + (1 - math.cos(Phi))* np.outer(e,e) 
    - math.sin(Phi)*Matrix_tilde(e)
    
    return C


def DCM_to_PR(C) :
    """
    This function returns the principal rotation angle Phi and principal rotation
    vector from the DCM matrix transformation from frame N to frame B

    Parameters
    ----------
    C : 3x3 ndarray. Orthogonal matrix. The direction cosine matrix [BN]

    Returns
    -------
    Phi : double. Angle of rotation in degrees
    e : 1d ndarray. Principal rotation vector

    """
    
    Phi = math.acos(0.5*(np.trace(C)-1))
    
    e = (1/(2*math.sin(Phi)))*np.array([C[1,2] - C[2,1], C[2,0] - C[0,2], C[0,1] - C[1,0]])
        
    return Phi*180/np.pi, e.reshape(-1)

def Add_PR(Phi1, e1, Phi2, e2):
    """
    This function computes the addition of two principal rotation parameters using eqs. (3.85) - (3.86) from Shaub's book.
    
    Parameters
    ----------
    Phi1 : double. First rotation angle in degrees
    e1 : 1d ndarray. First principal rotation vector
    Phi2 : double. Second rotation angle in degrees
    e2 : 1d ndarray. Second principal rotation vector

    Returns
    -------
    Phi : double. Angle of rotation in degrees
    e : 1d ndarray. Principal rotation vector
    
    """
    Phi1 = Phi1*np.pi/180  # in radians
    Phi2 = Phi2*np.pi/180  # in radians
    
    Phi = 2*math.acos(math.cos(Phi1/2)*math.cos(Phi2/2) -math.sin(Phi1/2)*math.sin(Phi2/2)*(e1.T @ e2)) # in radians
    e = (math.cos(Phi2/2)*math.sin(Phi1/2)*e1 + math.cos(Phi1/2)*math.sin(Phi2/2)*e2 + \
         math.sin(Phi1/2)*math.sin(Phi2/2)*(np.cross(e1,e2)))/math.sin(Phi/2)
    
    return Phi*180/np.pi, e 

def PR_to_EP(Phi, e) :
    """
    This function returns the Euler Parameters (Quaternions) set from the rotation angle Phi
    
    Parameters
    ---------
    Phi: double. Principal rotation angle in degrees.
    e: 1D ndarray. Principal rotation vector.
    
    Returns
    -------
    beta: tuple of four doubles, the EP set.
    
    """
    
    Phi = Phi*180/np.pi  # to radians
    b0 = math.cos(Phi/2)
    b1 = e[0]*math.sin(Phi/2)
    b2 = e[1]*math.sin(Phi/2)
    b3 = e[2]*math.sin(Phi/2)
    
    beta = (b0, b1, b2, b3)
    
    return beta

def DCM_to_EP(C) :
    """
    This function returns the EP from the DCM matrix using the Sheppard method explained in the book by Shaub- Junkins
    
    Parameters
    ----------
    C : 3x3 ndarray. The DCM matrix.
    
    Returns
    -------
    beta: tuple of 4 doubles, the EP set. 
    
    """
    
    # firs compute the above quantities:
    b0_sqr = (1+np.trace(C))/4
    b1_sqr = (1+2*C[0,0] -np.trace(C))/4
    b2_sqr = (1+2*C[1,1] -np.trace(C))/4
    b3_sqr = (1+2*C[2,2] -np.trace(C))/4   
    
    #check which one is the biggest:
    if max(b0_sqr,b1_sqr, b2_sqr, b3_sqr) == b0_sqr :
        b0 = np.sqrt(b0_sqr)
        b1 = (C[1,2]-C[2,1])/(4*b0)
        b2 = (C[2,0]-C[0,2])/(4*b0)
        b3 = (C[0,1]-C[1,0])/(4*b0)
    elif max(b0_sqr,b1_sqr, b2_sqr, b3_sqr) == b1_sqr :
        b1 = np.sqrt(b1_sqr)
        b0 = (C[1,2]-C[2,1])/(4*b1)
        b2 = (C[0,1]+C[1,0])/(4*b1)
        b3 = (C[2,0]+C[0,2])/(4*b1)
    elif max(b0_sqr,b1_sqr, b2_sqr, b3_sqr) == b2_sqr :
        b2 = np.sqrt(b2_sqr)
        b0 = (C[2,0]-C[0,2])/(4*b2)
        b1 = (C[0,1]+C[1,0])/(4*b2)
        b3 = (C[1,2]+C[2,1])/(4*b2)
    else :
        b3 = np.sqrt(b3_sqr)
        b0 = (C[0,1]-C[1,0])/(4*b3)
        b1 = (C[2,0]+C[0,2])/(4*b3) 
        b2 = (C[1,2]+C[2,1])/(4*b3)
    
    beta = np.array([b0, b1, b2, b3])
    if beta[0] < 0:
        beta = -1*beta
    
    return beta

def EP_to_DCM(beta):
    """
    This function returns the DCM matrix of the euler parameter beta.
    
    Parameters
    ----------
    beta: Tuple of doubles. The Euler parameters
    
    Returns
    -------
    C: 3x3 ndarray. The DCM matrix associated to the quaternions beta.
    
    """
    b0, b1, b2, b3 = beta
    C = np.array([[b0**2 + b1**2 -b2**2 - b3**2, 2*(b1*b2 + b0*b3), 2*(b1*b3 - b0*b2)],[2*(b1*b2 - b0*b3),b0**2-b1**2 +b2**2 - b3**2, 2*(b2*b3 + b0*b1)], [2*(b1*b3 + b0*b2), 2*(b2*b3 - b0*b1), b0**2-b1**2-b2**2+b3**2]])
    
    return C

def Add_EPs(beta1,beta2):
    """
    this function returns the composite rotation or addition of two euler parameters sets beta1, beta2: FN(beta)=FB(beta2)BN(beta1) then G(beta2)@beta1
    
    Parameters
    ----------
    beta1: 1d ndarray of 4 doubles representing the quaternions of the first set.
    beta2: 1d ndaray of 4 doubles representing the quaternions of the second set.
    
    Returns
    -------
    beta: 1d ndarray of four doubles representing the resultant quaternion by adding beta1 and beta2.
    
    """
    Gbeta2 = np.array([[beta2[0], -beta2[1], -beta2[2], -beta2[3]],\
                      [beta2[1],beta2[0], beta2[3], -beta2[2]],\
                      [beta2[2], -beta2[3], beta2[0], beta2[1]],\
                      [beta2[3], beta2[2], -beta2[1], beta2[0]]]) 
    
    beta = Gbeta2 @ beta1
    
    return beta

def Subtract_EPs(beta,beta1):
    """
    This function computes the subtraction of two EP's: knowing that beta= G(beta')beta''
    it returns beta'' from beta and beta'. The relation to the DCM's is:
    FN(beta) = FB(beta'')BN(beta')
    """
    Gbeta1 = np.array([[beta1[0], -beta1[1], -beta1[2], -beta1[3]],\
                      [beta1[1],beta1[0], -beta1[3], beta1[2]],\
                      [beta1[2], beta1[3], beta1[0], -beta1[1]],\
                      [beta1[3], -beta1[2], beta1[1], beta1[0]]]) 
    
    beta2 = Gbeta1.T @ beta
    
    return beta2


##################################################################
                # Classical Rodrigues Parameters #
##################################################################


def CRP_to_DCM(q):
    """
    This function computes the DCM matrix associated with the Classical Rodrigues Parameters CRP
    
    Parameters
    ----------
    q : 1d nd array. The CRP's
    
    Returns 
    -------
    C : 3x3 ndarray. An orthogonal matrix representing the DCM of the CRP orientation.
    
    """

    C = ((1 - q.T @ q)*np.eye(3) + 2*np.outer(q,q) - 2* Matrix_tilde(q))/(1 +q.T@q)
    
    return C

def DCM_to_CRP(C) :
    """
    This function computes the Classical Rodrigues Parameters from the DCM matrix.
    
    Parameters
    ----------
    C : 2D nd array. An orthogonal matrix representing the DCM
    
    Returns
    -------
    q : 2D nd array. Column vector representing the CRP set.
    
    """
    c = np.trace(C) + 1
    q = np.array([[C[1,2] - C[2,1]],[C[2,0] - C[0,2]], [C[0,1] - C[1,0]]])/c
    
    return q

def Add_CRPs(q1, q2):
    """
    This function adds two CRP's. Given two attitude vectors q1 q2, the overall composite attitude vector is q and is computed by:
    q = (q2 + q1 - q2xq1)/(1-q2^Tq1). In terms of the DCMs is 
    FN(q) = FB(q2)BN(q1). Since The associated DCM to a CRP q is such that C(q)^T = C(-q) then, to compute the subtraction, lets say q2 out of q1 and q we use the addition with a minus q1: i.e
    if FB(q2) = FN(q)BN(-q1) then 
    q2 = (q - q1 + qxq1)/(1 + q^Tq1)
    q1 = (-q2 + q + q2xq)/(1+ q2^Tq)
    
    Parameters
    ---------
    q1 : 2d ndarray vector column. Representing the firs CRP
    q2 : 2d ndarray vector column. Representing the firs CRP
    
    Returns
    -------
    q : 2d ndarray vector column. The overall composite of the two sequential CRP's
    """

    q = (q1 + q2 - np.cross(q2, q1))/(1 - q2.T @ q1)
    
    return q.reshape(-1)


##################################################################
                # Modified Rodrigues Parameters #
##################################################################



def MRP_to_DCM(sigma):
    """
    This function computes the DCM matrix associated to an MRP attitude.
    
    Parameters
    ----------
    sigma: 1d ndarray. The set of MRP's
    
    Returns
    -------
    C: 3x3 ndarray. The DCM matrix.
    
    """
    C = np.eye(3) +(8*Matrix_tilde(sigma)@Matrix_tilde(sigma) - 4*(1 - sigma.T@sigma)*Matrix_tilde(sigma))/((1+ sigma.T@sigma)**2)
    
    return C

def DCM_to_MRP(C, short = True):
    """
    This function computes the MRP set associated to the DCM matrix C.
    
    Parameters
    ----------
    C: 3x3 ndarray. The DCM matrix
    
    Returns
    -------
    sigma: 1d ndarray. The vector of the MRP's
    
    """
    
    s = np.sqrt(np.trace(C) + 1)
    sigma = np.array([C[1,2]-C[2,1], C[2,0]-C[0,2], C[0,1]-C[1,0]])/(s*(s + 2))
    
    if short :
        return sigma
    else :
        return -sigma/(sigma.T@sigma)
    

def Add_MRPs(sigma1, sigma2):
    """
    This function returns the overall sigma obtained from succesive rotation attitudes sigma1, sigma2 as in the DCM formalism:
    FN(sigma) = FB(sigma2)BN(sigma1) then:
    sigma = ((1-|\sigma_1|^2)\sigma_2 + (1-|\sigma_2|^2)\sigma_1 - 2\sigma_2\times \sigma_1)/(1+|\sigma_1|^2|\sigma_2|^2-2\sigma_1\cdot\sigma_2)
    
    """
    s1 = sigma1.T@sigma1 # |sigma1|^2
    s2 = sigma2.T@sigma2 # |sigma2|^2
    sigma = ((1 - s1) * sigma2 + (1 - s2) * sigma1 - 2*np.cross(sigma2, sigma1))/(1 + s1*s2 - 2*sigma1.T@sigma2)
    
    return sigma

def Subtract_MRPs(sigma1, sigma):
    """
    This function returns the subtraction sigma2 obtained from sigma1 and sigma as in the DCM formalism:
    FN(sigma)BN(sigma1)^T = FN(sigma)BN(-sigma1)= FB(sigma2) then:
    sigma2 = ((1-|\sigma_1|^2)\sigma - (1-|\sigma|^2)\sigma_1 + 2\sigma\times \sigma_1)/(1+|\sigma_1|^2|\sigma|^2+2\sigma_1\cdot\sigma).
    Obtained by replacing sigma1 in the Add_MRPs by -sigma1

    """
    
    return Add_MRPs(-sigma1, sigma)

######################################################################################
                        
                        # Kinematics Differential Equtions
        
######################################################################################




def EP_kinematics(t, beta0, w):
    """
    This function computes the rhs of the kinematic differential equation for the Euler Parameters 
    """
    w = (np.pi/180)*w  # angular velocity in rads/sec if given in deg/sec
    W = np.array([[0, -w[0], -w[1], -w[2]],[w[0], 0, w[2], -w[1]],[w[1], -w[2], 0, w[0]], [w[2], w[1], -w[0], 0]])
    rhs = 0.5*W@ beta0
    
    return rhs
    
def CRP_kinematics(t, q0, w):
    """
    this function computes the r.h.s of the kinematic differential equation of the Classical Rodrigues Parameter: q'=B(q)w 
    """
    w = (np.pi/180)*w  # angular velocity in rads/sec if given in deg/sec
    rhs = 0.5*(np.eye(3) + Matrix_tilde(q0) + np.outer(q0,q0))@w
    
    return rhs


def MRP_kinematics(t,sigma0, w):
    """
    this function computes the r.h.s of the kinematic differential equation of the Modified Rodrigues Parameter: sigma'=B(sigma)w 
    """
    B_sigma = 0.25*((1- norm(sigma0,2)**2)*np.eye(3) +2*Matrix_tilde(sigma0) +2*np.outer(sigma0,sigma0))
    w = w*np.pi/180  # angular velocity in rads/sec if given in deg/sec
    rhs = B_sigma @ w
    
    return rhs

## we write a special integrator for the MRP so we can check at every step time whether sigma is a long or a short attitude
def MRP_integrator(function, x0, t_span, step) :
    x = x0
    t0, t_f = t_span
    t = np.arange(t0, t_f, step)
    nt = t.shape[0]
    X = np.zeros((x0.shape[0],nt))
    X[:,0] = x
    
    for i in range(nt-1):
        k1 = function(t[i],x, w)
        k2 = function(t[i] + 0.5*step, x + 0.5*step*k1, w)
        k3 = function(t[i] + 0.5*step, x + 0.5*step*k2, w)
        k4 = function(t[i] + step, x + step*k3, w)
        x = x + step*(k1 +2*k2 + 2*k3 + k4)/6
        if norm(x,2) > 1 :
            x = -x/(norm(x,2)**2)
            #print('flag', i)
        X[:,i+1] = x
        
    return X,t



######################################################################################
                        
                        # Attitude Determination Methods
        
######################################################################################

def TRIAD(v_B1, v_B2, v_N1, v_N2) : 
    """
    This function computes the triad matrix or the DCM matrix of the body frame B relative to the innertial frame N if meassurements v_B, v_N in both frames are given by passing through a third frame T called the triad frame.
    The name “TRIAD” can be considered either as the word “triad” or as an acronym for TRIaxial Attitude Determination.
    Reference: Fundamentals of Spacecraft Attitude Determination and Control. Landis-Crassidis
    Parameters
    ----------
    v_B1, v_B2 : 1d ndarray. Measurements in the B frame
    v_N1, v_N2 : 1d ndarray. Measurements in the N frame
    
    Returns
    -------
    C : 3x3 ndarray. The DCM matrix corresponding to the attitude coordinates of the body relative to N, i.e the BN matrix.
    
    """
    
    v_B1 = v_B1/(norm(v_B1,2))
    v_B2 = v_B2/(norm(v_B2,2))
    v_N1 = v_N1/(norm(v_N1,2))
    v_N2 = v_N2/(norm(v_N2,2))

    ## T frame in both references
    t_B1 = v_B1
    t_B2 = np.cross(t_B1, v_B2)
    t_B2 = t_B2/norm(t_B2)
    t_B3 = np.cross(t_B1, t_B2)
    
    t_N1 = v_N1
    t_N2 = np.cross(t_N1, v_N2)
    t_N2 = t_N2/norm(t_N2)
    t_N3 = np.cross(t_N1, t_N2)
    
    C = np.outer(t_B1, t_N1) + np.outer(t_B3, t_N3) + np.outer(t_B2, t_N2)

    return C


def KB(B, Z) :
    KB =  np.vstack((np.hstack((np.trace(B), Z)), np.hstack((Z.reshape(-1,1), B + B.T - np.trace(B)*np.eye(3)))))
    return KB

def davenport(b, r, w):
    """
    this function applies the davenport method to estimate the DCM matrix barBN from measurements b and r relative to the body and to the inertial frame (or any other reference frame) respectively.
    
    Parameters
    ----------
    b : tuple. Measurements relative to the body frame.
    r : tuple. Measurements relative to another reference frame.
    w : tuple. Weights for each measurement.
    
    Returns
    -------
    barBN : 3x3 ndarray. The estimated DCM matrix.
    
    """
    B = np.zeros((3,3))
    Z = np.zeros(3)
    for i in range(len(b)) :
        B = B + w[i]*np.outer(b[i],r[i])
        Z = Z + w[i]*np.cross(b[i],r[i])
        
    K = KB(B, Z)
    alpha, v = eig(K)
    
    q = v[:,np.argmax(alpha)]
    
    if q[0] < 0 :
        q = -1*q
    print(q)
    barBN = EP_to_DCM(q)
    
    return barBN

#### QUEST Method ############################

def adjugate(A):
    """
    This function computes the adjugate matrix of A, i.e the transpose of the cofactor matrix.
    Parameters 
    ----------
    A : numpy array
    
    Returns
    -------
    C.T : numpy array.
    
    """
    
    U,sigma,Vt = np.linalg.svd(A)
    N = len(sigma)
    g = np.tile(sigma,N)
    g[::(N+1)] = 1
    G = np.diag(-(-1)**N*np.product(np.reshape(g,(N,N)),1)) 
    C = U @ G @ Vt
    
    return  C.T
        
def QUEST(b, r, w) :
    """
    This function implements the QUEST method for attitude determination.
    
    Parameters
    ----------
    b : tuple. Measurements relative to the body frame.
    r : tuple. Measurements relative to another reference frame.
    w : tuple. Weights for each measurement.
    


    """
    B = np.zeros((3,3))
    Z = np.zeros(3)
    for i in range(len(b)) :
        B = B + w[i]*np.outer(b[i],r[i])
        Z = Z + w[i]*np.cross(b[i],r[i])
        

    S = B + B.T
    adjS = adjugate(S)
    k = np.trace(adjS)
    a = np.trace(B)
    aa = det(S)
    z = Z@Z  # norm squared of Z

    f = lambda t: (t**2 -a**2 + k)*(t**2 - a**2 - z) - (t - a)*(Z@(S@Z) + aa) - Z@(S@S@Z)
    fprime = lambda t : t*(4*t**2 - 4*a**2 - 2*z + 2*k) - Z@(S@Z) - aa
    x0 = np.sum(w)  # initial guess for the eignevalue we want to estimate
    
    # Newton method to compute an estimation of the largest eignevalue:
    x = optimize.root_scalar(f, x0 = 1.9, fprime = fprime, method='newton')
    l_max = x.root
    
    # compute the quaternion :
    q0 = det((l_max +a)*np.eye(3) - S)
    q = np.hstack((q0, (adjugate((l_max + a)*np.eye(3) - S))@Z ) )
    
    q = q/(norm(q,2))
    
    if q[0] < 0:
        q = -1*q
        
    return q, EP_to_DCM(q)
    