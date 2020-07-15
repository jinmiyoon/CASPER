July 15 2020

First, I recognize that mods1 and mods2 are different in terms of normalization around Ca K line. The mods1 spectra look better than mod2. 

 It appears that flux_min=80 in GISIC.normalize() fit looks better than flux_min=70. I tried to flux_min=90 but it does not appear to be much different from 80 one.


Now I want to try different sigma value in batch.normalize() to  see gaussian filter size makes different. chaning the value will make # of continuum points changed.

smaller simga makes more continuum points.

I tried two different sigma ranges
  
  1) sigma = [15, 25] : best fit
  2) sigma = [25, 35]  : not recommend due to too high continuum around CN band below CH.
  3) sigma = [10,20]  : not recommend due to make too false continuum even at C2 band
 
at this current flux_min=80, the first sigma choice [15, 25] looks much better than [25,35]. So I will stick with flux_min=80 and sigma [15, 25]. both of the choices  give more consistent results between mods1 and mods2 than before using the first default setup [15,30].

