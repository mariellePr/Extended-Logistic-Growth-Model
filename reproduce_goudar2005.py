#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:49:19 2025

@author: mpere
"""

# =============================================================================
# IMPORT
# =============================================================================
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =============================================================================
# FUNCTIONS
# =============================================================================
def get_data():
    
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
                              'ylabel':'Viable cells density\n 10^6 cells/mL',
                              'xlabel':'Time (h)'})


     # obtained with: https://automeris.io/wpd/ from goudar2005 figure 5
    ref_14_time=[0.06593988819996532,
        0.88248924419287,
        1.445553651609095,
        1.955286692979131, 
        2.4510162267656117,
        2.8693240691993243, 
        3.982790708592236, 
        4.923386461264003, 
        5.4346217946353415, 
        6.491028164621693, 
        7.027265929157541 ]
     
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
                               'ylabel':'Viable cells density\n 10^6 cells/mL',
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
                               'ylabel':'Viable cells density\n 10^6 cells/mL',
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

    
    dict_data = {'Ljumggren, 1994 (ref_15_figure_6 Goudar 2005)':ref_15_df,
                 ' Linz, 1997 (ref_14_figure_5 Goudar 2005)':ref_14_df,
                 'Bayley, 2012 (generation age 100 - Figure 1A)':ref_bayley_2012_100_df,
                 'Wright, 2009 (Figure 7 - C. jejuni strains 11168H)':ref_wright_2009_square_df,
                 'Wright, 2009 (Figure 7 - 11168H cj0688)':ref_wright_2009_dot_df,
                 'Torres, 2020 (Figure 1 - 5 mL of working vol.)': ref_torres_2020_5ml_df,
                 'Torres, 2020 (Figure 1 - 10 mL of working vol.)': ref_torres_2020_10ml_df}
    
    return dict_data
    
def goudar_model():
    pass
    
def fit_goudar_mode():
    pass

# Define the integro-differential equation model
def integro_differential_model_1(t, y, a, b, m):
    x = y[0]  # Current value of x
    integral = y[1]  # Current value of the integral
    dx_dt = a * x * np.exp(-b * integral) - m * x
    dintegral_dt = x  # The derivative of the integral is simply x
    return [dx_dt, dintegral_dt]

# Define a function to solve the integro-differential equation
def solve_model_1(t_eval, a, b, m, x0, integral0):
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

# Define a function for curve fitting
def fit_model_1(t, a, b, m, x0):
    return solve_model_1(t, a, b, m, x0=x0, integral0=0.0)


# Define the integro-differential equation model
def integro_differential_model_2(t, y, a, b, c, m):
    x = y[0]  # Current value of x
    integral = y[1]  # Current value of the integral
    dx_dt = a * x * 1 / (1 + c * np.exp(b * integral)) - m * x
    dintegral_dt = x  # The derivative of the integral is simply x
    return [dx_dt, dintegral_dt]

# Define a function to solve the integro-differential equation
def solve_model_2(t_eval, a, b,c, m, x0, integral0):
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

# Define a function for curve fitting
def fit_model_2(t, a, b, c,m, x0):
    return solve_model_2(t, a, b,c, m, x0=x0, integral0=0.0)
  

# =============================================================================
# MAIN
# =============================================================================
if __name__=='__main__': 
    dict_data = get_data()
    
    
    fig, axs = plt.subplots(len(dict_data.keys())//4 +1,4, figsize =(16,11)) 
    for (label, data_df), ax in zip(dict_data.items(), axs.flat):
    
        ref_15_time = data_df['Time'].to_numpy()
        ref_15_data = data_df['Data'].to_numpy()
        
        if 'Wright' in label:
            ref_15_data =  ref_15_data*1e-9
        
        
        # Fit the model_1
        popt_ref_15, pcov_ref_15 = curve_fit(
            lambda t, a, b, m, x0: fit_model_1(t, a, b, m, x0),
            ref_15_time,
            ref_15_data,
            p0=[0.4, 0.02, 0.01, 0.36],  # Initial guess for a, b, m, x0
            bounds=(0, [2, 1, 1, 1])  # Bounds for a, b, m, x0
        )
        
        # Extract fitted parameters for T=30 and T=25
        a_fit_ref_15, b_fit_ref_15, m_fit_ref_15, x0_fit_ref_15 = popt_ref_15
        
        # Solve the model with fitted parameters
        t_fine = np.linspace(0, ref_15_time[-1], 500)
        x_fit_ref_15 = solve_model_1(t_fine, a_fit_ref_15, b_fit_ref_15, m_fit_ref_15, 
                                   x0=x0_fit_ref_15, integral0=0.0)
        
        ax.scatter(ref_15_time,ref_15_data,color = 'k', marker = 'o',
                         label = 'Data')
        ax.plot(t_fine,x_fit_ref_15,'-b', label = 'cmvd model 1' + r'$a \cdot x \cdot e^{-b \cdot \int} - m \cdot x$')
        
        
        # Fit the model_2
        popt_ref_15, pcov_ref_15 = curve_fit(
            lambda t, a, b, c,m, x0: fit_model_2(t, a, b,c, m, x0),
            ref_15_time,
            ref_15_data,
            p0=[0.4, 0.02, 0.5, 0.01, 0.36],  # Initial guess for a, b, c, m, x0
            bounds=(0, [5, 5, 5, 5, 5])  # Bounds for a, b, c, m, x0 # Bounds for a, b, m, x0
        )
        
      
        
        # Extract fitted parameters for T=30 and T=25
        a_fit_ref_15, b_fit_ref_15, c_fit_ref_15, m_fit_ref_15, x0_fit_ref_15 = popt_ref_15
        
        # Solve the model with fitted parameters
        t_fine = np.linspace(0, ref_15_time[-1], 500)
        x_fit_ref_15 = solve_model_2(t_fine, a_fit_ref_15, b_fit_ref_15,c_fit_ref_15, m_fit_ref_15, 
                                   x0=x0_fit_ref_15, integral0=0.0)
        
       
        ax.plot(t_fine,x_fit_ref_15,'-r', label = 'cmvd model 2' + r'$a \cdot x \cdot \frac{1}{1 + c \cdot e^{b \cdot \int}} - m \cdot x$')
        
        
        # Add labels and title
        ylabel =  data_df['ylabel'].drop_duplicates().to_list()[0]
        xlabel =  data_df['xlabel'].drop_duplicates().to_list()[0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend()
        ax.grid()
        
    # Show the plot
    fig.tight_layout()
    fig.show()