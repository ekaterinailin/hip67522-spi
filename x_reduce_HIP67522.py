# -*- coding: utf-8 -*-
"""
@author: Alexis Brandeker, alexis@astro.su.se

Script used to reduce the CHEOPS light curves with PIPE.
"""

if __name__ == '__main__': # This line is to break loops when multiprocessing
    from pipe import PipeParam, PipeControl, conf

    
    visits = ['visit_103','visit_104',
              'visit_501','visit_601', 'visit_701','visit_801',
              'visit_901','visit_1001', ]

    conf.data_root = '/data/minoss-vdb/ilin_cheops'
    conf.ref_lib_data = '/home/ilin/cal/Ref'
        
    for visit in visits:
        pps = PipeParam('', "CHEOPS-products-" + visit)

        pps.klip = 1
        pps.fitrad = 30
        pps.bgstars = True
        pps.fit_bgstars = False
        pps.nthreads = 30

        pps.remove_satellites = False

        pps.sa_optimise = True
        pps.im_optimise = True

        pps.sa_test_klips = [1,3,5]
        pps.sa_test_fitrads = [30,60,50,25,40]

        pps.im_test_klips = [1,3,5]
        pps.im_test_fitrads = [30,25]
        
        pc = PipeControl(pps)
        pc.process_eigen()
