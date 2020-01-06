# -*- coding: utf-8 -*-
"""
This is an easy numerical and categorical binning solution 
for general score build. It is designed to choose the optimal 
binning solution by utilizing the Tree and Target Encoder. 
There are numerical solutution as well as categorical one. 

@author: Tri Le <lmtleminh@gmail.com>

"""
from ._classes import ScBinning
from ._classes import ScCatBinning

__all__ = ["ScBinning", "ScCatBinning"]