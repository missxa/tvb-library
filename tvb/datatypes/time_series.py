# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#

"""
The TimeSeries datatypes. This brings together the scientific and framework 
methods that are associated with the time-series data.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import tvb.datatypes.time_series_scientific as time_series_scientific
import tvb.datatypes.time_series_framework as time_series_framework
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)



class TimeSeries(time_series_scientific.TimeSeriesScientific,
                 time_series_framework.TimeSeriesFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the TimeSeries datatype.
    
    ::
        
                           TimeSeriesData
                                 |
                                / \\
             TimeSeriesFramework   TimeSeriesScientific
                                \ /
                                 |
                             TimeSeries
        
    
    """
    pass



class TimeSeriesEEG(time_series_scientific.TimeSeriesEEGScientific,
                    time_series_framework.TimeSeriesEEGFramework, TimeSeries):
    """
    This class brings together the scientific and framework methods that are
    associated with the TimeSeriesEEG datatype.
    
    ::
        
                         TimeSeriesEEGData
                                 |
                                / \\
          TimeSeriesEEGFramework   TimeSeriesEEGScientific
                                \ /
                                 |
                           TimeSeriesEEG
        
    
    """
    pass


class TimeSeriesMEG(time_series_scientific.TimeSeriesMEGScientific,
                    time_series_framework.TimeSeriesMEGFramework, TimeSeries):
    """
    This class brings together the scientific and framework methods that are
    associated with the TimeSeriesMEG datatype.
    
    ::
        
                         TimeSeriesMEGData
                                 |
                                / \\
          TimeSeriesMEGFramework   TimeSeriesMEGScientific
                                \ /
                                 |
                           TimeSeriesMEG
        
    
    """
    pass


class TimeSeriesRegion(time_series_scientific.TimeSeriesRegionScientific,
                       time_series_framework.TimeSeriesRegionFramework, TimeSeries):
    """
    This class brings together the scientific and framework methods that are
    associated with the TimeSeriesRegion dataType.
    
    ::
        
                         TimeSeriesRegionData
                                  |
                                 / \\
        TimeSeriesRegionFramework   TimeSeriesRegionScientific
                                 \ /
                                  |
                           TimeSeriesRegion
        
    
    """
    pass


class TimeSeriesSurface(time_series_scientific.TimeSeriesSurfaceScientific,
                        time_series_framework.TimeSeriesSurfaceFramework, TimeSeries):
    """
    This class brings together the scientific and framework methods that are
    associated with the TimeSeriesSurface dataType.
    
    ::
        
                         TimeSeriesSurfaceData
                                   |
                                  / \\
        TimeSeriesSurfaceFramework   TimeSeriesSurfaceScientific
                                  \ /
                                   |
                            TimeSeriesSurface
        
    
    """
    pass


class TimeSeriesVolume(time_series_scientific.TimeSeriesVolumeScientific,
                       time_series_framework.TimeSeriesVolumeFramework, TimeSeries):
    """
    This class brings together the scientific and framework methods that are
    associated with the TimeSeriesVolume dataType.
    
    ::
        
                         TimeSeriesVolumeData
                                  |
                                 / \\
        TimeSeriesVolumeFramework   TimeSeriesVolumeScientific
                                 \ /
                                  |
                           TimeSeriesVolume
        
    
    """
    pass



if __name__ == '__main__':
    # Do some stuff that tests or makes use of this module...
    LOG.info("Testing %s module..." % __file__)
    
    # Check that all default TimeSeries datatypes initialize without error.
    TIME_SERIES = TimeSeries()
    TIME_SERIES_EEG = TimeSeriesEEG()
    TIME_SERIES_MEG = TimeSeriesMEG()
    TIME_SERIES_REGION = TimeSeriesRegion()
    TIME_SERIES_SURFACE = TimeSeriesSurface()
    TIME_SERIES_VOLUME = TimeSeriesVolume()
    
    LOG.info("Default TimeSeries datatypes initialized without error...")
