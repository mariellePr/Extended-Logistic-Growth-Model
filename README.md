# Extended Logistic Growth Model
This Python code is associated to the submitted manuscript "A Single Population Approach to Modeling Growth and Decay in Batch Bioreactors", Carlos Matinez von Dossow, Marielle Péré. 

DOI: [10.5281/zenodo.15240209](https://doi.org/10.5281/zenodo.15240208).

It runs on Python 3.12 and uses Matplotlib, Scipy, Numpy and Pandas libraries.

Its purpose is to calibrate the two new models of cell proliferation presented in our associated paper using data from different studies on batch culture of different cell lines such as CHO.
It organizes data from 6 different papers:
 * *logistic equations effectively model mammalian cell batch and fed-batch kinetics
by logically constraining the fit*,  Goudar et al., 2005 (with data from Ljumggren, 1994 and Linz, 1997),
 * *Determination of Chinese hamster ovary cell line stability and recombinant antibody expression during long-term culture*, Bayley et al., 2012,
 * *Metabolite and transcriptome analysis of Campylobacter jejuni in vitro growth reveals a stationary-phase physiological switch*, Wright et al., 2009,
 * *Metabolic profiling of Chinese hamster ovary cell cultures at different working volumes and agitation speeds using spin tube reactors*, Torres et al., 2021,
 * *Influence of yeast extract concentrationon batch cultures of Lactobacillus helveticus: growth and production coupling*, Amrane et al., 1998.
and presents the corresponding calibration of model M1 and M2 results.

Therefore two figures are generated:
 * Figure 3 and 4 of the paper 
![Figure3](images/Figure_3.png)  

![Figure4](images/Figure_4.png)
 
The code is organized in three parts: Import, Functions and Main with a function per figure named accordingly. 
