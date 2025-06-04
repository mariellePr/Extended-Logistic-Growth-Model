#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:25:11 2025

@author: Marielle Péré, Carlos Martinez Von Dossow

This code is associated to the submitted manuscript "A Single Population Approach to Modeling Growth and Decay in Batch Bioreactors", Carlos Matinez von Dossow, Marielle Péré.

Its purpose is to calibrate the models presented in this paper to data from different studies on batch culture.
"""

# =============================================================================
# IMPORT
# =============================================================================
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# FUNCTIONS
# =============================================================================
def compute_calibration_metrics(t_eval,X_data, model_to_fit, popt,pcov,names  = ['a','b','c','m','x0']):
    """
    Compute different metrics to evaluate quality of fit.

    Parameters
    ----------
    t_eval : numpy vector
        Time vector
    X_data : numpy vector
        Data vector
    model_to_fit : Python function
        Model calibrated.
    popt : list
        List of estimated parameters.
    pcov : numpy matrix
        Covariance matrix.
    names : List of str,
        List of parameters names. The default is ['a','b','c','m','x0'].

    Returns
    -------
    str
        Legend for plotting model containing AIC and RMSE.

    """
    
    fitted = model_to_fit(t_eval, *popt)
    resid   = X_data - fitted
    N, P    = X_data.size, popt.size
    dof     = N - P
    σ2      = np.sum(resid**2) / dof
    σ       = np.sqrt(σ2)
    pcov    = pcov * σ2
    
    # Compute R²
    ss_res = np.sum((X_data - fitted) ** 2)
    ss_tot = np.sum((X_data - np.mean(X_data)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
  
    
    # 7) Standard errors and 95 % t‐intervals
    perr   = np.sqrt(np.diag(pcov))
    alpha  = 0.05
    tval   = stats.t.ppf(1 - alpha/2, dof)
    ci95   = tval * perr
    
    # AIC/RMSE
    aic_1, rmse_1, sigma_1 = compute_aic_and_rmse(X_data, fitted, len(names))
    
   
    print(f"\nEstimation of\nσ = {σ:.4f},\ndof = {dof},\nt₀.₉₇₅ = {tval:.3f}")
    print(f'AIC = {aic_1},\nRMSE = {rmse_1},\nR² = {r2:.3f}')
    print("\nParam.   Value    stderr    IC 95%")
    for nm, pv, err, ci in zip(names, popt, perr, ci95):
        lo, hi = pv - ci, pv + ci
        print(f"{nm:>4s}     {pv:7.4f}  {err:7.4f}   [{lo:7.4f}, {hi:7.4f}]")
    print('------------')
    return f'AIC = {aic_1:.3f}\nRMSE = {rmse_1:.3f}',aic_1, rmse_1,r2

def compute_aic_and_rmse(data, sol, k):
    """
    Computes the AIC, RMSE, and estimates sigma between observed data and model solution.
    
    Parameters:
    - data: Observed data vector
    - sol: Model solution vector (must be the same length as data)
    - k: Number of parameters in the model
    
    - AIC value
    - RMSE value
    - Estimated sigma
    """
    # Ensure data and solution are numpy arrays
    data = np.asarray(data)
    sol = np.asarray(sol)

    # Compute residuals
    residuals = data - sol
    
    # Number of data points
    n = len(data)
    
    # print('k = ',k, 'n = ',n, 'sol =',sol,'data = ', data)
    # Estimate variance of residuals (degrees of freedom: n - k)
    variance = np.sum(residuals ** 2) / (n - k)
    
    # Estimate standard deviation (sigma)
    sigma = np.sqrt(variance)
    
    # Compute log-likelihood assuming Gaussian residuals
    log_likelihood = -0.5 * np.sum((residuals / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

    # AIC formula: 2k - 2 * log-likelihood
    aic = 2 * k - 2 * log_likelihood
    
    # RMSE calculation: root mean squared error
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    return aic, rmse, sigma
def get_data():
    """
    Organize data from 6 papers on batch culture into a python dict for plotting.
        - Ljumggren, 1994 (from Goudar, 2005),
        - Linz, 1997 (from Goudar, 2005),
        - Bayley, 2012,
        - Wright, 2009,
        - Torres, 2020,
        - Amrane, 1998.
        

    Returns
    -------
    Dict organized as follow:
            key: Reference of the paper and additional info. Will be used as title during plotting.
            value: dict
                key/value: Time, Data, xlabel, ylabel

    """
    
    # obtained with: https://automeris.io/wpd/ from goudar2005 figure 6
    ref_15_time=[   0.588302674824007, 
    11.277612483882635, 
    18.840077422159318, 
    24.695201647130013, 
    34.263711153079136, 
    46.271488614301965, 
    57.113821735184494,
    65.6628611541071, 
    71.91772573875602,
    82.29805181645429 ,
    90.97397428855908]
    
    ref_15_data=[  0.16040191381938573,
     0.20714774532926472,
    0.31335238537202903,
     0.43691885652522555,
    0.6420429346405423,
     0.9955602679817079,
     1.1313744782962427,
     1.1484825433003107,
     1.0394618043827524,
    0.8833286829127888,
     0.7692995879237884]
    ref_15_df = pd.DataFrame({'Time':ref_15_time,'Data':ref_15_data,
                              'ylabel':'Viable cells density\n'+r"($10^6$ cells/mL)",
                              'xlabel':'Time (h)'})


     # obtained with: https://automeris.io/wpd/ from goudar2005 figure 5
    ref_14_time=[0.06593988819996532*24,
        0.88248924419287*24,
        1.445553651609095*24,
        1.955286692979131*24, 
        2.4510162267656117*24,
        2.8693240691993243*24, 
        3.982790708592236*24, 
        4.923386461264003*24, 
        5.4346217946353415*24, 
        6.491028164621693*24, 
        7.027265929157541 *24]
     
    ref_14_data=[ 0.27997793508623103,
         0.342359939774187,
         0.9122835844418886,
        1.0319471783401688,
         1.5026994309398383,
        1.7076315092333956,
         1.5212601144826987,
         1.6862590581836354,
         1.6073384281599272,
         1.4636213955890294,
         1.079691896006519 ]
    ref_14_df = pd.DataFrame({'Time':ref_14_time,'Data':ref_14_data,
                               'ylabel':'Viable cells density\n'+r"($10^6$ cells/mL)",
                               'xlabel':'Time (days)'})
    
    # Obtained from Figure 1A - https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/bit.24485
    
    ref_bayley_2012_100_time=[0, 
        3.0191256830601088,
        5.027322404371584, 
        7.0081967213114735, 
        9.002732240437158,
        10.997267759562842, 
        12.978142076502731, 
        14.972677595628415]
     
    ref_bayley_2012_100_data=[ 0.2089552238805954,
        1.2835820895522376,
         3.7711442786069647,
         5.5621890547263675,
         6.199004975124378,
         4.965174129353233,
         2.6368159203980097,
        1.1044776119402977 ]
    ref_bayley_2012_100_df = pd.DataFrame({'Time':ref_bayley_2012_100_time,'Data':ref_bayley_2012_100_data,
                               'ylabel':r'Viable cells density\n $10^6$ cells/mL',
                               'xlabel':'Time (days)'})
    
    
    # from Figure 7 - https://www.microbiologyresearch.org/docserver/fulltext/micro/155/1/80.pdf?expires=1738003547&id=id&accname=guest&checksum=8DE49E6B41D30269FA30F6B1E8006FDF
    ref_wright_2009_square_time=[0, 
        7.912087912087912,
        15.897435897435894, 
        19.999999999999996, 
        27.83882783882783, 
        39.92673992673993, 
        53.84615384615384, 
        65.86080586080585]
     
    ref_wright_2009_square_data=[ 275790197.8947369,
        635512221.0526316,
         786810829.4736842,
         978245635.7894738,
         969549545.2631578,
         897546532.6315789,
         777058252.6315789,
         580847408.4210526]
    
    
    
    # cfu = colony forming unit
    ref_wright_2009_square_df = pd.DataFrame({'Time':ref_wright_2009_square_time,'Data':ref_wright_2009_square_data,
                               'ylabel':'Viable cells density\n c.f.u/mL - mean val. across 2 exp',
                               'xlabel':'Time (h)'})
    
    ref_wright_2009_dot_time=[0, 
        7.912087912087912,
        15.897435897435894, 
        19.999999999999996, 
        27.83882783882783, 
        39.92673992673993, 
        53.84615384615384, 
        65.86080586080585]
     
    ref_wright_2009_dot_data=[  269479562.0743792,
        626807733.8926781,
         889409453.0176299,
         1007020968.4004636,
         970670470.714194,
        746673783.1167164,
         528848753.22609735,
         443799295.10230374
         ]
    
    ref_wright_2009_dot_df = pd.DataFrame({'Time':ref_wright_2009_dot_time,'Data':ref_wright_2009_dot_data,
                               'ylabel':'Viable cells density\n c.f.u/mL - mean val. across 2 rep.',
                               'xlabel':'Time (h)'})
    
    
    
    # from https://doi.org/10.1002/btpr.3099
#     Figure 1. Cell growth, viability and r-protein production profiles of the CHO-IgG1 cell line during
# 10 days batch cultures at different working volume. 5 mL (white dots), 10 mL (light grey dots), 15
# mL (dark grey dots) and 20 mL (black dots). Experimental values represent the mean of three biological
# replicates and error bars indicate ± SEM.
    ref_torres_2020_5ml_time=[0,
            3,
            4,
            5, 
            6,
            7,
            10]
     
    ref_torres_2020_5ml_data=[  1.7977528089887613,
             30.112359550561806,
            47.47191011235956,
             52.528089887640455,
             38.876404494382015,
            14.438202247191015,
            -0.05617977528089568 ]
    
    ref_torres_2020_5ml_df = pd.DataFrame({'Time':ref_torres_2020_5ml_time,'Data':ref_torres_2020_5ml_data,
                               'ylabel':'Viable cells density\n (%) - mean val. across 3 rep.',
                               'xlabel':'Time (days)'})
    
    ref_torres_2020_10ml_time=[0,
        3,
        4,
        5, 
        6,
        7,
        10]
     
    ref_torres_2020_10ml_data=[ 1.62921348314606,
         26.57303370786518,
        53.37078651685393,
        68.53932584269664,
         65.84269662921349,
         50.67415730337079,
         -0.05617977528089568 ]
    
    ref_torres_2020_10ml_df = pd.DataFrame({'Time':ref_torres_2020_10ml_time,'Data':ref_torres_2020_10ml_data,
                               'ylabel':'Viable cells density\n (%) - mean val. across 3 rep.',
                               'xlabel':'Time (days)'})
    
    
    # Data for S0=5 g/L
    data_time_5 = np.array([0.09, 0.59, 1.10, 1.55, 2.53, 3.11, 3.66, 4.05, 5.02, 6.03, 7.02, 8.04, 9.02, 10.04, 11.06, 12.01, 12.98, 14.03, 15.02])
    data_x_5 = np.array([0.23, 0.26, 0.27, 0.34, 0.60, 0.88, 1.26, 1.49, 1.77, 1.85, 1.88, 1.88, 1.93, 1.93, 1.91, 1.91, 1.91, 1.88, 1.89])
    
    ref_amrane_1998_5gl_df = pd.DataFrame({'Time':data_time_5,'Data':data_x_5,
                               'ylabel':'Biomass\n ()g/L',
                               'xlabel':'Time (h)'})
    
    # Data for D0=10 g/L
    data_time_10 = np.array([0.00, 0.53, 1.06, 1.54, 2.06, 2.55, 3.05, 3.57, 4.06, 4.54, 5.02, 5.51, 6.02, 6.53, 7.02, 7.54, 8.07, 8.53, 9.04, 9.52, 10.03, 10.53, 11.01, 11.54, 12.03, 12.52, 13.03, 14.02, 15.02])
    data_x_10 = np.array([0.27, 0.32, 0.42, 0.52, 0.67, 0.97, 1.41, 2.03, 2.80, 3.60, 4.19, 4.48, 4.57, 4.66, 4.71, 4.73, 4.74, 4.74, 4.76, 4.76, 4.75, 4.77, 4.76, 4.68, 4.61, 4.57, 4.53, 4.50, 4.53])
    ref_amrane_1998_10gl_df = pd.DataFrame({'Time':data_time_10,'Data':data_x_10,
                               'ylabel':'Biomass\n ()g/L',
                               'xlabel':'Time (h)'})
    
    
    # Data for S0=20 g/L
    data_time_20 = np.array([0.00, 1.12, 1.99, 2.55, 3.05, 3.55, 4.02, 4.54, 5.04, 5.49, 6.01, 6.51, 6.98, 7.52, 8.00, 9.02, 10.06, 10.98, 12.02, 13.02, 13.99, 15.08])
    data_x_20 = np.array([0.26, 0.24, 0.45, 0.86, 1.31, 1.89, 2.76, 3.76, 4.78, 5.61, 6.12, 6.23, 6.29, 6.15, 6.01, 5.89, 5.78, 5.66, 5.61, 5.57, 5.51, 5.37])
    ref_amrane_1998_20gl_df = pd.DataFrame({'Time':data_time_20,'Data':data_x_20,
                               'ylabel':'Biomass\n ()g/L',
                               'xlabel':'Time (h)'})
    
    # Data for S0=2 g/L
    data_time_2 = np.array([0.00, 1.06, 2.04, 3.06, 4.03, 5.06, 6.07, 7.04, 8.04, 9.06, 10.06, 11.03, 12.03, 13.02, 14.03, 15.04])
    data_x_2 = np.array([0.10, 0.13, 0.17, 0.30, 0.53, 0.74, 0.81, 0.85, 0.83, 0.87, 0.89, 0.88, 0.90, 0.91, 0.91, 0.93])
    ref_amrane_1998_2gl_df = pd.DataFrame({'Time':data_time_2,'Data':data_x_2,
                               'ylabel':'Biomass\n ()g/L',
                               'xlabel':'Time (h)'})

    
    dict_data = {'Ljumggren, 1994 (ref_15_figure_6 Goudar 2005)':ref_15_df,
                 ' Linz, 1997 (ref_14_figure_5 Goudar 2005)':ref_14_df,
                 'Bayley, 2012 (generation age 100 - Figure 1A)':ref_bayley_2012_100_df,
                 'Wright, 2009 (Figure 7 - C. jejuni strains 11168H)':ref_wright_2009_square_df,
                 'Wright, 2009 (Figure 7 - 11168H cj0688)':ref_wright_2009_dot_df,
                 'Torres, 2021 (Figure 1 - 5 mL of working vol.)': ref_torres_2020_5ml_df,
                 'Torres, 2021 (Figure 1 - 10 mL of working vol.)': ref_torres_2020_10ml_df,
                 'Amrane, 1998 (Figure 1 - 2 g/L)':ref_amrane_1998_2gl_df,
                 'Amrane, 1998 (Figure 1 - 5 g/L)':ref_amrane_1998_5gl_df,
                 'Amrane, 1998 (Figure 1 - 10 g/L)':ref_amrane_1998_10gl_df,
                 'Amrane, 1998 (Figure 1 - 20 g/L)':ref_amrane_1998_20gl_df}
    
    return dict_data

    
# Define the integro-differential equation model
def integro_differential_model_1(t, y, a, b, m):
    """
    Model M1 in the paper.

    Parameters
    ----------
    t : double
       Time point.
    y : double
        State variable, Biomass.
    a : Positive double
        Growth rate.
    b : Positive double
        Combination of parameter.
    m : Positive double
        Maintenance Rate.

    Returns
    -------
    list
        [Biomass, Cumulative Biomass].

    """
    x = y[0]  # Current value of x
    integral = y[1]  # Current value of the integral
    dx_dt = a * x * np.exp(-b * integral) - m * x
    dintegral_dt = x  # The derivative of the integral is simply x
    return [dx_dt, dintegral_dt]

# Define a function to solve the integro-differential equation
def solve_model_1(t_eval, a, b, m, x0, integral0):
    """
    Solve model M1 using given parameters a,b and m, initial condition x0 and integral0 for time t_eval.

    Parameters
    ----------
    t_eval : numpy array
        Time vector
    a : Positive double
        Growth rate.
    b : Positive double
        Combination of parameter.
    m : Positive double
        Maintenance Rate.
    x0 : double
        Initial condition for biomass.
    integral0 : double
        Initial condition for cumulative biomass.

    Returns
    -------
    numpy vector
        Biomass value, solution of model M1 at time t_eval.

    """
    t_span = (t_eval[0], t_eval[-1])
    y0 = [x0, integral0]  # Initial conditions: [x, integral]

    sol = solve_ivp(
        lambda t, y: integro_differential_model_1(t, y, a, b, m),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45'
    )
    return sol.y[0]  # Return only the population (x)

def solve_model1_full(t_eval, a, b, m, x0, integral0=0.0):
    """
    Solve model M1 using given parameters a,b and m, initial condition x0 and integral0 for time t_eval.

    Parameters
    ----------
    t_eval : numpy array
        Time vector
    a : Positive double
        Growth rate.
    b : Positive double
        Combination of parameter.
    m : Positive double
        Maintenance Rate.
    x0 : double
        Initial condition for biomass.
    integral0 : double
        Initial condition for cumulative biomass.

    Returns
    -------
    numpy vector
        Biomass value, solution of model M1 at time t_eval.

    """
    t_span = (t_eval[0], t_eval[-1])
    y0 = [x0, integral0]  # Initial conditions: [x, integral]

    sol = solve_ivp(
        lambda t, y: integro_differential_model_1(t, y, a, b, m),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45'
    )
    return sol.y  # Return only the population (x)

# Define a function for curve fitting
def solve_model_1_with_null_cumulative_biomass(t, a, b, m, x0):
    """
    Solve model M1 when a,b,m and x0 are given and cumulative biomass is equal to 0

    Parameters
    ----------
   t_eval : numpy array
       Time vector
   a : Positive double
       Growth rate.
   b : Positive double
       Combination of parameter.
   m : Positive double
       Maintenance Rate.
   x0 : double
       Initial condition for biomass.

    Returns
    -------
    numpy array
        Biomass value, solution of model M1 at time t_eval.

    """
    return solve_model_1(t, a, b, m, x0=x0, integral0=0.0)


# Define the integro-differential equation model
def integro_differential_model_2(t, y, a, b, c, m):
    """
    Model M2 in the paper.

    Parameters
    ----------
    t : double
       Time point.
    y : double
        State variable, Biomass.
    a : Positive double
        Growth rate.
    b : Positive double
        Combination of parameter.
    c: Positive double
        Dimensionless parameter.        
    m : Positive double
        Maintenance Rate.

    Returns
    -------
    list
        [Biomass, Cumulative Biomass].

    """
    x = y[0]  # Current value of x
    integral = y[1]  # Current value of the integral
    dx_dt = a * x * 1 / (1 + c * np.exp(b * integral)) - m * x
    dintegral_dt = x  # The derivative of the integral is simply x
    return [dx_dt, dintegral_dt]

# Define a function to solve the integro-differential equation
def solve_model_2(t_eval, a, b,c, m, x0, integral0):
    """
    Solve model M2 using given parameters a,b and m, initial condition x0 and integral0 for time t_eval.

    Parameters
    ----------
    t_eval : numpy array
        Time vector
    a : Positive double
        Growth rate.
    b : Positive double
        Combination of parameter.
    c: Positive double
         Dimensionless parameter.   
    m : Positive double
        Maintenance Rate.
    x0 : double
        Initial condition for biomass.
    integral0 : double
        Initial condition for cumulative biomass.

    Returns
    -------
    numpy vector
        Biomass value, solution of model M2 at time t_eval.

    """
    t_span = (t_eval[0], t_eval[-1])
    y0 = [x0, integral0]  # Initial conditions: [x, integral]

    sol = solve_ivp(
        lambda t, y: integro_differential_model_2(t, y, a, b,c, m),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45'
    )
    return sol.y[0]  # Return only the population (x)

def solve_model2_full(t_eval, a, b,c, m, x0, integral0=0.0):
    """
    Solve model M2 using given parameters a,b and m, initial condition x0 and integral0 for time t_eval.

    Parameters
    ----------
    t_eval : numpy array
        Time vector
    a : Positive double
        Growth rate.
    b : Positive double
        Combination of parameter.
    c: Positive double
         Dimensionless parameter.   
    m : Positive double
        Maintenance Rate.
    x0 : double
        Initial condition for biomass.
    integral0 : double
        Initial condition for cumulative biomass.

    Returns
    -------
    numpy vector
        Biomass value, solution of model M2 at time t_eval.

    """
    t_span = (t_eval[0], t_eval[-1])
    y0 = [x0, integral0]  # Initial conditions: [x, integral]

    sol = solve_ivp(
        lambda t, y: integro_differential_model_2(t, y, a, b,c, m),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45'
    )
    return sol.y  # Return only the population (x)

# Define a function for curve fitting
def solve_model_2_with_null_cumulative_biomass(t, a, b, c,m, x0):
    """
    Solve model M2 when a,b,m,c and x0 are given and cumulative biomass is equal to 0

    Parameters
    ----------
   t_eval : numpy array
       Time vector
   a : Positive double
       Growth rate.
   b : Positive double
       Combination of parameter.
   c: Positive double
        Dimensionless parameter. 
   m : Positive double
       Maintenance Rate.
   x0 : double
       Initial condition for biomass.

    Returns
    -------
    numpy array
        Biomass value, solution of model M2 at time t_eval.

    """
    return solve_model_2(t, a, b,c, m, x0=x0, integral0=0.0)


def Figure_3():
    """
    Create Figure 3 of the  paper. Plot data from Goudar 2005 and calibration of model M1 and M2 results in the same figure

    Returns
    -------
    Matplotlib figure.

    """
    dict_data = get_data()
    fig, axs = plt.subplots(2,2, figsize =(8,11*2/3), sharex='col') 
    for i,((label, data_df)) in enumerate(dict_data.items()):
        if i < 2:
            print(label,'\n')
        
            ref_15_time = data_df['Time'].to_numpy()
            ref_15_data = data_df['Data'].to_numpy()
            
            if 'Wright' in label:
                ref_15_data =  ref_15_data*1e-9
            
            
            # Fit the model_1
            popt_ref_15, pcov_ref_15 = curve_fit(
                lambda t, a, b, m, x0: solve_model_1_with_null_cumulative_biomass(t, a, b, m, x0),
                ref_15_time,
                ref_15_data,
                p0=[0.4, 0.02, 0.01, 0.36],  # Initial guess for a, b, m, x0
                bounds=(0, [5, 1, 1, 1])  # Bounds for a, b, m, x0
            )
            
            print('Metrics for Model 1')
            legend_model_1,_,_,_ = compute_calibration_metrics(t_eval = ref_15_time,
                                        X_data = ref_15_data, 
                                        model_to_fit = solve_model_1_with_null_cumulative_biomass,
                                        popt = popt_ref_15,
                                        pcov = pcov_ref_15,
                                        names  = ['a','b','m','x0'])
            
            
            # Extract fitted parameters for T=30 and T=25
            a_fit_ref_15, b_fit_ref_15, m_fit_ref_15, x0_fit_ref_15 = popt_ref_15
            
            # Solve the model with fitted parameters
            t_fine = np.linspace(0, ref_15_time[-1], 500)
            x_fit_ref_15 = solve_model_1(t_fine, a_fit_ref_15, b_fit_ref_15, m_fit_ref_15, 
                                       x0=x0_fit_ref_15, integral0=0.0)
            
            x_fit_short = solve_model_1(ref_15_time, a_fit_ref_15, b_fit_ref_15, m_fit_ref_15, 
                                       x0=x0_fit_ref_15, integral0=0.0)
            
            
            axs[0,i].scatter(ref_15_time,ref_15_data,color = 'k', marker = 'o',
                             label = '_Data')
            axs[1,i].scatter(ref_15_time,ref_15_data,color = 'k', marker = 'o',
                             label = '_Data')
            
            aic_1, rmse_1, sigma_1 = compute_aic_and_rmse(ref_15_data, x_fit_short, 3)
            from matplotlib.patches import Patch
            axs[0,i].plot(t_fine,x_fit_ref_15,'-b',label =   legend_model_1)
            from matplotlib.lines import Line2D
            custom_legend = [Line2D([0], [0], color='none', label=legend_model_1)]

            axs[0,i].legend(handles=custom_legend)
            
            
            # Fit the model_2
            popt_ref_15, pcov_ref_15 = curve_fit(
                lambda t, a, b, c,m, x0: solve_model_2_with_null_cumulative_biomass(t, a, b,c, m, x0),
                ref_15_time,
                ref_15_data,
                p0=[0.4, 0.02, 0.5, 0.01, 0.36],  # Initial guess for a, b, c, m, x0
                bounds=(0, [10, 5, 5, 5, 5])  # Bounds for a, b, c, m, x0 # Bounds for a, b, m, x0
            )
            
            print('Metrics for Model 2')
            legend_model_2,_,_,_ = compute_calibration_metrics(t_eval = ref_15_time,
                                        X_data = ref_15_data, 
                                        model_to_fit = solve_model_2_with_null_cumulative_biomass,
                                        popt = popt_ref_15,
                                        pcov = pcov_ref_15,
                                        names  = ['a','b','c','m','x0'])
            
            
            # Extract fitted parameters for T=30 and T=25
            a_fit_ref_15, b_fit_ref_15, c_fit_ref_15, m_fit_ref_15, x0_fit_ref_15 = popt_ref_15
            
            # Solve the model with fitted parameters
            t_fine = np.linspace(0, ref_15_time[-1], 500)
            x_fit_ref_15 = solve_model_2(t_fine, a_fit_ref_15, b_fit_ref_15,c_fit_ref_15, m_fit_ref_15, 
                                       x0=x0_fit_ref_15, integral0=0.0)
            
            x_fit_short = solve_model_2(ref_15_time, a_fit_ref_15, b_fit_ref_15,c_fit_ref_15, m_fit_ref_15, 
                                       x0=x0_fit_ref_15, integral0=0.0)
            
            aic_2, rmse_2, sigma_2 = compute_aic_and_rmse(ref_15_data, x_fit_short, 4)
            
           
            axs[1,i].plot(t_fine,x_fit_ref_15,'r', label = legend_model_2)
            
            
            # Add labels and title
            ylabel =  data_df['ylabel'].drop_duplicates().to_list()[0]
            xlabel =  data_df['xlabel'].drop_duplicates().to_list()[0]
            axs[0,0].set_title('Model 1', fontweight = 'bold')
            axs[0,1].set_title('Model 2', fontweight = 'bold')
            
            
            
            
            for row in range(2):
                axs[row,i].set_xlabel('Time (h)')
                axs[row,i].set_ylabel(ylabel)
                
                
                if i==0:
                    axs[row,i].set_xticks([0,24,48,72,24*4])  
                    max_y = max(line.get_ydata().max() for line in axs[0,0].lines)
                    axs[0,0].text(-0.1,max_y-0.05*max_y,'A', fontweight = 'bold', fontsize = 12)
                    axs[1,0].text(-0.1,max_y-0.05*max_y,'C', fontweight = 'bold', fontsize = 12)
                    
                    
                else:
                    axs[row,i].set_xticks([0,24,48,72,24*4,24*5,24*6,24*7,24*8])  
                    # axs[row,i].set_xticks([0,2,4,6,8],[0,48,24*4,24*6,24*8])
                    max_y = max(line.get_ydata().max() for line in axs[0,1].lines)
                    axs[0,1].text(-0.10,max_y-0.05*max_y,'B', fontweight = 'bold', fontsize = 12)
                    axs[1,1].text(-0.10,max_y-0.05*max_y,'D', fontweight = 'bold', fontsize = 12)
                    
                    
                # ax.set_title(label)
                axs[row,i].legend(title = 'Metrics', loc = 'lower right', title_fontproperties={'weight':'bold'})
                axs[row,i].grid()
            
            # print('Model 2:',f'a = {a_fit_ref_15:.3f}, b={b_fit_ref_15:.3f}, c={c_fit_ref_15:.3f}, m={m_fit_ref_15:.3f}\n AIC ={aic_2:.3f}')
            print('\n')
    # Show the plot
    fig.tight_layout()
    fig.show()
   
    

def Figure_5():
    """
    Create Figure 5 of the  paper. Plot data from Amrane 1998 and calibration of model M2 in the same figure

    Returns
    -------
    Matplotlib figure.

    """

    dict_data = get_data()
    data_time_2 = dict_data['Amrane, 1998 (Figure 1 - 2 g/L)']['Time']
    data_x_2 = dict_data['Amrane, 1998 (Figure 1 - 2 g/L)']['Data']
    
    data_time_5 = dict_data['Amrane, 1998 (Figure 1 - 5 g/L)']['Time']
    data_x_5 = dict_data['Amrane, 1998 (Figure 1 - 5 g/L)']['Data']
    
    data_time_10 = dict_data['Amrane, 1998 (Figure 1 - 10 g/L)']['Time']
    data_x_10 = dict_data['Amrane, 1998 (Figure 1 - 10 g/L)']['Data']
    
    data_time_20 = dict_data['Amrane, 1998 (Figure 1 - 20 g/L)']['Time']
    data_x_20 = dict_data['Amrane, 1998 (Figure 1 - 20 g/L)']['Data']
    
    
    # Define a function for curve fitting with shared parameters (a, b, m) and independent (c, x0)
    def fit_model(t, a, b, m, c1, x0_1, c2, x0_2, c3, x0_3, c4, x0_4):
        t_5 = t[:len(data_time_5)]
        t_10 = t[len(data_time_5):len(data_time_5) + len(data_time_10)]
        t_20 = t[len(data_time_5) + len(data_time_10):len(data_time_5) + len(data_time_10) + len(data_time_20)]
        t_2 = t[len(data_time_5) + len(data_time_10) + len(data_time_20):]
    
        x_5 = solve_model_2(t_5, a, b, c4, m, x0_4, integral0=0.0)
        x_10 = solve_model_2(t_10, a, b, c1, m, x0_1, integral0=0.0)
        x_20 = solve_model_2(t_20, a, b, c2, m, x0_2, integral0=0.0)
        x_2 = solve_model_2(t_2, a, b, c3, m, x0_3, integral0=0.0)
    
        return np.concatenate([x_5, x_10, x_20, x_2])
    
    # Combine data for fitting
    combined_time = np.concatenate([data_time_5, data_time_10, data_time_20, data_time_2])
    combined_x = np.concatenate([data_x_5, data_x_10, data_x_20, data_x_2])
    
    # Perform the curve fitting
    popt, pcov = curve_fit(
        lambda t, a, b, m, c1, x0_1, c2, x0_2, c3, x0_3, c4, x0_4: fit_model(t, a, b, m, c1, x0_1, c2, x0_2, c3, x0_3, c4, x0_4),
        combined_time,
        combined_x,
        p0=[0.4, 0.02, 0.01, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3],  # Initial guesses
        bounds=(0, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])  # Bounds for the parameters
    )
    
    print('Metrics for Model 2')
    legend_model_1,_,_,_ = compute_calibration_metrics(t_eval = combined_time,
                                X_data = combined_x, 
                                model_to_fit = fit_model,
                                popt = popt,
                                pcov = pcov,
                                names  = ['a','b','m','c1', 'x0_1', 'c2', 'x0_2', 'c3', 'x0_3', 'c4', 'x0_4'])
    
    # Extract fitted parameters
    a_fit, b_fit, m_fit, c1_fit, x0_1_fit, c2_fit, x0_2_fit, c3_fit, x0_3_fit, c4_fit, x0_4_fit = popt
    # Perform the curve fitting
    
    # Solve the model with fitted parameters
    t_fine_5 = np.linspace(0, max(data_time_5), 500)
    t_fine_10 = np.linspace(0, max(data_time_10), 500)
    t_fine_20 = np.linspace(0, max(data_time_20), 500)
    t_fine_2 = np.linspace(0, max(data_time_2), 500)
    x_fit_5 = solve_model_2(t_fine_5, a_fit, b_fit, c4_fit, m_fit, x0_4_fit, integral0=0.0)
    x_fit_10 = solve_model_2(t_fine_10, a_fit, b_fit, c1_fit, m_fit, x0_1_fit, integral0=0.0)
    x_fit_20 = solve_model_2(t_fine_20, a_fit, b_fit, c2_fit, m_fit, x0_2_fit, integral0=0.0)
    x_fit_2 = solve_model_2(t_fine_2, a_fit, b_fit, c3_fit, m_fit, x0_3_fit, integral0=0.0)
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot the original data points
    ax.scatter(data_time_5, data_x_5, color='purple')
    ax.scatter(data_time_10, data_x_10, color='red')
    ax.scatter(data_time_20, data_x_20, color='blue')
    ax.scatter(data_time_2, data_x_2, color='green')
    
    # Plot the fitted models
    ax.plot(t_fine_20, x_fit_20, color='blue', label='c = 0.021')
    ax.plot(t_fine_10, x_fit_10, color='red', label='c = 0.047')
    ax.plot(t_fine_5, x_fit_5, color='purple', label='c = 0.43')
    ax.plot(t_fine_2, x_fit_2, color='green', label='c = 1.29')
    
    # Labels and legend
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('x (g/L)')
    ax.legend()
    ax.grid(True)
    
    # Adjust layout before saving
    #plt.tight_layout()
    
    # Save the figure as an EPS file
    #filename = "/content/fitted_4p.eps"  # Define filename
    #plt.savefig(filename, format='eps', dpi=300)  # Save as EPS
    
    # Close the figure to ensure proper saving
    #plt.close()
    
    # Download the EPS file
    #google.colab.files.download(filename)
    
    # Print fitted parameters
    print(f"\nFitted parameters:\na = {a_fit:.4f} h-1,\nb = {b_fit:.4f} g-1 L h-1,\nm = {m_fit:.4f} h-1")
    print(f"\nS0=5 g/L: c = {c4_fit:.4f}, x0 = {x0_4_fit:.4f} g/L")
    print(f"S0=10 g/L: c = {c1_fit:.4f}, x0 = {x0_1_fit:.4f} g/L")
    print(f"S0=20 g/L: c = {c2_fit:.4f}, x0 = {x0_2_fit:.4f} g/L")
    print(f"S0=2 g/L: c = {c3_fit:.4f}, x0 = {x0_3_fit:.4f} g/L")
    
def supplementary_figure_all_data():
    """
    Plot data from 6 papers on batch culture
        - Ljumggren, 1994 (from Goudar, 2005),
        - Linz, 1997 (from Goudar, 2005),
        - Bayley, 2012,
        - Wright, 2009,
        - Torres, 2020,
        - Amrane, 1998,
    and the corresponding calibration of model M1 and M2 results in the same figure.

    Returns
    -------
    Matplotlib figure.

    """
    dict_data = get_data()
    fig, axs = plt.subplots(len(dict_data.keys())//4 +1,4, figsize =(16,11)) 
    for (label, data_df), ax in zip(dict_data.items(), axs.flat):
        print('--------------------------------------------')
        print('Data from: ', label)
        ref_15_time = data_df['Time'].to_numpy()
        ref_15_data = data_df['Data'].to_numpy()
        
        if 'Wright' in label:
            ref_15_data =  ref_15_data*1e-9
        
        
        # Fit the model_1
        popt_ref_15, pcov_ref_15 = curve_fit(
            lambda t, a, b, m, x0: solve_model_1_with_null_cumulative_biomass(t, a, b, m, x0),
            ref_15_time,
            ref_15_data,
            p0=[0.4, 0.02, 0.01, 0.36],  # Initial guess for a, b, m, x0
            bounds=(0, [5, 1, 1, 1])  # Bounds for a, b, m, x0
        )
        print('Metrics for Model 1')
        legend_model_1,_,_,_  = compute_calibration_metrics(t_eval = ref_15_time,
                                    X_data = ref_15_data, 
                                    model_to_fit = solve_model_1_with_null_cumulative_biomass,
                                    popt = popt_ref_15,
                                    pcov = pcov_ref_15,
                                    names  = ['a','b','m','x0'])
        
        # Extract fitted parameters for T=30 and T=25
        a_fit_ref_15, b_fit_ref_15, m_fit_ref_15, x0_fit_ref_15 = popt_ref_15
        
        # Solve the model with fitted parameters
        t_fine = np.linspace(0, ref_15_time[-1], 500)
        x_fit_ref_15 = solve_model_1(t_fine, a_fit_ref_15, b_fit_ref_15, m_fit_ref_15, 
                                   x0=x0_fit_ref_15, integral0=0.0)
        
        x_fit_short = solve_model_1(ref_15_time, a_fit_ref_15, b_fit_ref_15, m_fit_ref_15, 
                                   x0=x0_fit_ref_15, integral0=0.0)
        
        
        ax.scatter(ref_15_time,ref_15_data,color = 'k', marker = 'o',
                         label = 'Data')
        
        aic_1, rmse_1, sigma_1 = compute_aic_and_rmse(ref_15_data, x_fit_short, 4)
        ax.plot(t_fine,x_fit_ref_15,'-b', label =  f'Model 1:\na = {a_fit_ref_15:.3f}, b={b_fit_ref_15:.3f},\nm={m_fit_ref_15:.3f}\nAIC ={aic_1:.3f}')
        
        
        # Fit the model_2
        popt_ref_15, pcov_ref_15 = curve_fit(
            lambda t, a, b, c,m, x0: solve_model_2_with_null_cumulative_biomass(t, a, b,c, m, x0),
            ref_15_time,
            ref_15_data,
            p0=[0.4, 0.02, 0.5, 0.01, 0.36],  # Initial guess for a, b, c, m, x0
            bounds=(0, [10, 5, 5, 5, 5])  # Bounds for a, b, c, m, x0 # Bounds for a, b, m, x0
        )
        
        print('Metrics for Model 2')
        legend_model_1,_,_,_  = compute_calibration_metrics(t_eval = ref_15_time,
                                    X_data = ref_15_data, 
                                    model_to_fit = solve_model_2_with_null_cumulative_biomass,
                                    popt = popt_ref_15,
                                    pcov = pcov_ref_15,
                                    names  = ['a','b','c','m','x0'])
       
        
        # Extract fitted parameters for T=30 and T=25
        a_fit_ref_15, b_fit_ref_15, c_fit_ref_15, m_fit_ref_15, x0_fit_ref_15 = popt_ref_15
        
        # Solve the model with fitted parameters
        t_fine = np.linspace(0, ref_15_time[-1], 500)
        x_fit_ref_15 = solve_model_2(t_fine, a_fit_ref_15, b_fit_ref_15,c_fit_ref_15, m_fit_ref_15, 
                                   x0=x0_fit_ref_15, integral0=0.0)
        
        x_fit_short = solve_model_2(ref_15_time, a_fit_ref_15, b_fit_ref_15,c_fit_ref_15, m_fit_ref_15, 
                                   x0=x0_fit_ref_15, integral0=0.0)
        
        aic_2, rmse_2, sigma_2 = compute_aic_and_rmse(ref_15_data, x_fit_short, 5)
        
       
        ax.plot(t_fine,x_fit_ref_15,'-r', label = f'Model 2:\na = {a_fit_ref_15:.3f}, b={b_fit_ref_15:.3f},\nc={c_fit_ref_15:.3f}, m={m_fit_ref_15:.3f}\nAIC ={aic_2:.3f}')
        
        
        # Add labels and title
        ylabel =  data_df['ylabel'].drop_duplicates().to_list()[0]
        xlabel =  data_df['xlabel'].drop_duplicates().to_list()[0]
        ax.set_xlabel(xlabel, fontsize = 9)
        ax.set_ylabel(ylabel, fontsize = 9)
        ax.set_title(label, fontweight = 'bold', fontsize =10 ) 
        ax.legend(loc = 'lower right')
        ax.grid()
        
    # Show the plot
    fig.tight_layout()
    fig.show()
    
def Figure_4():
    dict_data = get_data()
    
    exp_dict = dict_data['Amrane, 1998 (Figure 1 - 10 g/L)']
    
    # —————————— Experimental data ——————————
    t_data = np.array(exp_dict['Time'])
    x_data = np.array(exp_dict['Data'])
    # ————— Fit for Model 1 —————
    popt1, pcov1 = curve_fit(
        lambda t, a, b, m, x0: solve_model1_full(t, a, b, m, x0)[0],
        t_data, x_data,
        p0=[0.4, 0.02, 0.01, 0.27],
        bounds=(0, [5, 1, 1, 1])
    )
    _,aic1, rmse1, r21 = compute_calibration_metrics(
        t_data, x_data,
        lambda t, a, b, m, x0: solve_model1_full(t, a, b, m, x0)[0],
        popt1, pcov1,
        ['a','b','m','x0']
    )
    
    # ————— Fit for Model 2 —————
    popt2, pcov2 = curve_fit(
        lambda t, a, b, c, m, x0: solve_model2_full(t, a, b, c, m, x0)[0],
        t_data, x_data,
        p0=[0.4, 0.02, 0.5, 0.01, 0.27],
        bounds=(0, [10, 5, 5, 5, 5])
    )
    _,aic2, rmse2, r22 = compute_calibration_metrics(
        t_data, x_data,
        lambda t, a, b, c, m, x0: solve_model2_full(t, a, b, c, m, x0)[0],
        popt2, pcov2,
        ['a','b','c','m','x0']
    )
    # ————— Prepare smooth curves for plotting —————
    t_fine = np.linspace(t_data.min(), t_data.max(), 300)
    
    # Model 1: solve on t_fine to obtain x(t) and X(t)
    sol1_fine = solve_model1_full(t_fine, *popt1)
    x_fit1 = sol1_fine[0]   # biomass
    X_fit1 = sol1_fine[1]   # accumulated X
    growth_term1 = popt1[0] * np.exp(-popt1[1] * X_fit1)  # growth rate
    
    # Model 2: solve on t_fine to obtain x(t) and X(t)
    sol2_fine = solve_model2_full(t_fine, *popt2)
    x_fit2 = sol2_fine[0]
    X_fit2 = sol2_fine[1]
    growth_term2 = popt2[0] / (1 + popt2[2] * np.exp(popt2[1] * X_fit2))  # growth rate
    
    # ————— Plot only the linear‐scale subplots —————
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    
    # -------- Left: Model 1 --------
    ax1 = axes[0]
    ax1.scatter(t_data, x_data, color='k', label='Experimental data')
    ax1.plot(
        t_fine,
        x_fit1,
        'b-',
        label=f'Model 1 fit\n(AIC={aic1:.2f}, RMSE={rmse1:.3f})'
    )
    ax1.set_title('Model 1')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Biomass (g/L)')
    ax1.grid(True)
    
    ax1b = ax1.twinx()
    ax1b.plot(t_fine, growth_term1, 'r--', label='Growth rate')
    ax1b.set_ylabel('Growth rate')
    
    # -------- Right: Model 2 --------
    ax2 = axes[1]
    ax2.scatter(t_data, x_data, color='k', label='Experimental data')
    ax2.plot(
        t_fine,
        x_fit2,
        'g-',
        label=f'Model 2 fit\n(AIC={aic2:.2f}, RMSE={rmse2:.3f})'
    )
    ax2.set_title('Model 2')
    ax2.set_xlabel('Time (h)')
    ax2.grid(True)
    
    ax2b = ax2.twinx()
    ax2b.plot(t_fine, growth_term2, 'r--', label='Growth rate')
    ax2b.set_ylabel('Growth rate')
    
    plt.tight_layout()
    
    # Save figure as SVG
    #filename = 'model_fits_linear.svg'
    #plt.savefig(filename, format='svg', dpi=300)
    # If running in Colab or Jupyter, you can download or display as needed
    # For example: from google.colab import files; files.download(filename)
    
    plt.show()
# =============================================================================
# MAIN
# =============================================================================
if __name__=='__main__': 
    plt.close('all')
    print('\n\nGenerate Figure 3')
    Figure_3()
    
    print('\n\nGenerate Figure 4')
    Figure_4()
    
    print('\n\nGenerate Figure 5')
    Figure_5()
    print('\n\nModels fit on all data')
    supplementary_figure_all_data()
