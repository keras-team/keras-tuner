from math import log, ceil
import numpy as np
from random import random
from time import time, ctime

from ..engine import HyperTuner
class HyperBand(HyperTuner):
    
    def __init__(self, 
                eta=3,
                skip_last = 0,
                **kwargs):
        
		super(HyperBand, self).__init__(**kwargs)
		self.eta = eta
		self.logeta = lambda x: log( x ) / log( self.eta )
		self.s_max = int( self.logeta( self.iterations ))
		self.B = ( self.s_max + 1 ) * self.iterations

		self.results = []	# list of dicts
		self.counter = 0
		self.best_loss = np.inf
		self.best_counter = -1

		def search(self):
			for s in reversed( range( self.s_max + 1 )):
				
				# initial number of configurations
				n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))	

				# initial number of iterations per config
				r = self.max_iter * self.eta ** ( -s )		

				# n random configurations
				T = [ self.get_params() for i in range( n )] 

				for i in range(( s + 1 ) - int( skip_last )):	# changed from s + 1
					
					# Run each of the n configs for <iterations> 
					# and keep best (n_configs / eta) configurations
					
					n_configs = n * self.eta ** ( -i )
					n_iterations = r * self.eta ** ( i )
					
					print "\n*** {} configurations x {:.1f} iterations each".format( 
						n_configs, n_iterations )
					
					val_losses = []
					early_stops = []
					
					for t in T:
						
						self.counter += 1
						print "\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format( 
							self.counter, ctime(), self.best_loss, self.best_counter )
						
						start_time = time()
						
						if dry_run:
							result = { 'loss': random(), 'log_loss': random(), 'auc': random()}
						else:
							result = self.try_params( n_iterations, t )		# <---
							
						assert( type( result ) == dict )
						assert( 'loss' in result )
						
						seconds = int( round( time() - start_time ))
						print "\n{} seconds.".format( seconds )
						
						loss = result['loss']	
						val_losses.append( loss )
						
						early_stop = result.get( 'early_stop', False )
						early_stops.append( early_stop )
						
						# keeping track of the best result so far (for display only)
						# could do it be checking results each time, but hey
						if loss < self.best_loss:
							self.best_loss = loss
							self.best_counter = self.counter
						
						result['counter'] = self.counter
						result['seconds'] = seconds
						result['params'] = t
						result['iterations'] = n_iterations
						
						self.results.append( result )
					
					# select a number of best configurations for the next loop
					# filter out early stops, if any
					indices = np.argsort( val_losses )
					T = [ T[i] for i in indices if not early_stops[i]]
					T = T[ 0:int( n_configs / self.eta )]
			