#! /usr/bin/env python3
###############################################################################
# isce2he5.py
#
#  Project:  Seamless SAR Archive
#  Purpose:  Create HDF-EOS5 interferogram product 
#  Author:   Scott Baker
#  Created:  April 2015
#
###############################################################################
#  Copyright (c) 2015, Scott Baker
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
###############################################################################

import os
import sys
import glob
import argparse
import datetime
import numpy as np
import h5py
import isce
import isceobj
import pickle
import logging
from imageMath import IML


fzero = np.float32(0.0)
czero = np.complex64(0.0)

def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description='Create HDF-EOS5 interferogram product from ISCE output')
    parser.add_argument('--radar', dest='rdrcoord', action='store_true', default=False,
            help='Store product in radar coordinates')
    parser.add_argument('--geo', dest='geocoord', action='store_true', default=False,
            help='Store product in geo coordinates')
    parser.add_argument('-o', '--out', dest='outhe5', type=str, required=True,
            help='Output he5 file')

    inps = parser.parse_args()

    if (inps.rdrcoord is False) and (inps.geocoord is False):
        raise Exception('Specify Radar / Geo coordinates')

    if inps.rdrcoord:
        raise NotImplementedError('Currently, the script only works for geocoded products')

    return inps

def main(argv):
    inps = cmdLineParse()

    ####Default names
    int_file = "filt_topophase.flat"
    cor_file = "phsig.cor"
    unw_file = "filt_topophase.unw"
    rdr_file = 'los.rdr'

    if inps.geocoord:
        int_file  += '.geo'
        cor_file  += '.geo'
        unw_file  += '.geo'
        rdr_file  += '.geo'

    print(unw_file)
    ####Gather some basic information
    unw = IML.mmapFromISCE(unw_file, logging)
    shape = unw.bands[0].shape


    h5file = inps.outhe5

    ## OPEN HDF5 FILE ##
    f = h5py.File(h5file)
    hdfeos = f.create_group('HDFEOS')

    if inps.geocoord:
        ## CREATE GRIDS GROUP ##
        group = hdfeos.create_group('GRIDS')
    else:
        ## CREATE SWATHS GROUP ##
        group = hdfeos.create_group('SWATHS')

    
    insar = group.create_group('InSAR')
    data  = group.create_group('Data Fields')

    ## CREATE UNWRAPPED INTERFEROGRAM ##
    dset = data.create_dataset('UnwrappedInterferogram', data=unw.bands[1],
        shape=shape, chunks=(128,128), compression='gzip')
    dset.attrs['Title'] = 'Unwrapped phase'
    dset.attrs['MissingValue'] = fzero
    dset.attrs['Units'] = 'radians'
    dset.attrs['_FillValue'] = fzero
    unw = None

    #### CREATE COHERENCE ####
    cor = IML.mmapFromISCE(cor_file, logging)
    dset = data.create_dataset('Coherence', data=cor.bands[0],
        shape=shape, chunks=(128,128), compression='gzip')
    dset.attrs['Title'] = 'Phase Sigma Coherence'
    dset.attrs['MissingValue'] = fzero
    dset.attrs['Units'] = 'None'
    dset.attrs['_FillValue'] = fzero
    cor = None


    #### CREATE WRAPPED INTERFEROGRAM
    wrap = IML.mmapFromISCE(int_file, logging)
    dset = data.create_dataset('WrappedInterferogram', data=wrap.bands[0],
        shape=shape, chunks=(64,64), compression='gzip')
    dset.attrs['Title'] = 'Wrapped interferogram'
    dset.attrs['MissingValue'] = czero
    dset.attrs['Units'] = 'None'
    dset.attrs['_FillValue'] = czero
    wrap = None


    #### CREATE ILLUMINATION ANGLE
    ang = IML.mmapFromISCE(rdr_file, logging)
    dset = data.create_dataset('Illimunation', data=ang.bands[0],
        shape=shape, chunks=(128,128), compression='gzip')
    dset.attrs['Title'] = 'Illumination angle'
    dset.attrs['Description'] = 'Vertical angle of the vector from target to sensor, w.r.t to the normal of the ellipse at the target'
    dset.attrs['MissingValue'] = fzero
    dset.attrs['Units'] = 'degrees'
    dset.attrs['_FillValue'] = fzero


    #### CREATE AZIMUTH ANGLE
    dset = data.create_dataset('Azimuth', data=360.0-ang.bands[1],
        shape=shape, chunks=(128,128), compression='gzip')
    dset.attrs['Title'] = 'Azimuth angle'
    dset.attrs['Description'] = 'Angle of the vector from target to sensor, measured clockwise w.r.t North at the target'
    dset.attrs['MissingValue'] = fzero
    dset.attrs['Units'] = 'degrees'
    dset.attrs['_FillValue'] = fzero
    ang = None


    ## WRITE ATTRIBUTES TO THE HDF ##
    if inps.geocoord:
        geoinfo = IML.getGeoInfo(unw_file)
        if geoinfo is None:
            raise Exception('No geocoding information found')

        north = geoinfo[0]
        south = geoinfo[0] + (shape[1] - 1) * geoinfo[2]
        west = geoinfo[1]
        east = geoinfo[1] + (shape[0] - 1) * geoinfo[3]

        insar.attrs['GCTPProjectionCode'] = np.zeros(1, dtype=np.int32)
        insar.attrs['GCTPSpheroidCode'] = str(12)
        insar.attrs['Projection'] = 'Geographic'
        insar.attrs['GridOrigin'] = 'Center'
        insar.attrs['GridSpacing'] = str((geoinfo[-1], geoinfo[-2]))
        insar.attrs['GridSpacingUnit'] = 'deg'
        insar.attrs['GridSpan'] = str((west,east,north,south))
        insar.attrs['GridSpanUnit'] = 'deg'
        insar.attrs['NumberOfLongitudesInGrid'] = np.array([shape[0]], dtype=np.int32)
        insar.attrs['NumberOfLatitudesInGrid'] = np.array([shape[1]], dtype=np.int32)

    f.close()

if __name__ == '__main__':
    main(sys.argv[:])

