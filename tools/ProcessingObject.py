import argparse
'''
Class containing some helper functions, i.e., argument parsing
'''
class ProcessingObject(object):
    def __init__(self):
        '''
        Constructor
        '''
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("num_samples", type=int,\
                            help="total # of samples")
        parser.add_argument("--num_rfx", type=int,\
                            help="number of random features of the data X",
                            default=30)
        parser.add_argument("--num_rfy", type=int,\
                            help="number of random features of the data Y",
                            default=30)
        parser.add_argument("--num_inducex", type=int,\
                            help="number of inducing variables of the data X",
                            default=30)
        parser.add_argument("--num_inducey", type=int,\
                            help="number of inducing variables of the data Y",
                            default=30)
        parser.add_argument("--num_shuffles", type=int,\
                            help="number of shuffles",
                            default=800)
        parser.add_argument("--blocksize", type=int,\
                            help="# of samples per block (includes X and Y) when using a block-based test",
                            default=20)
        parser.add_argument("--dimX", type=int,\
                            help="dimensionality of the data X",
                            default=3)
        parser.add_argument("--kernel_width_x", action="store_true",\
                            help="should median heuristic be used for X?",
                            default=False)
        parser.add_argument("--kernel_width_y", action="store_true",\
                            help="should median heuristic be used for Y?",
                            default=False)
        #parser.add_argument("--dimY", type=int,\
        #                    help="dimensionality of the data Y",
        #                    default=3)
        parser.add_argument("--hypothesis", type=str,\
                            help="is null or alternative true in this experiment? [null, alter]",\
                            default="alter")
        parser.add_argument("--nullvarmethod", type=str,\
                            help="how to estimate asymptotic variance under null? [direct, permutation, across]?",\
                            default="direct")
        parser.add_argument("--streaming", action="store_true",\
                            help="should data be streamed (rather than all loaded into memory)?",\
                            default=False)
        parser.add_argument("--rff", action="store_true",\
                            help="should random features be used?",\
                            default=False)
        parser.add_argument("--induce_set", action="store_true",\
                            help="should inducing variables be used?",\
                            default=False)
        args = parser.parse_args()
        return args