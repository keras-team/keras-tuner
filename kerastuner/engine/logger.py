"Logging functions to ensure unified output accross tuners"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from termcolor import cprint

SETTING_COLORS = ['green', 'cyan']



# TODO need to refactor to make it part of python logging system
class Logger():
 
    def __init__(self, hypertuner):
        
        # store a reference to the current tuner
        self.hypertuner = hypertuner

    def tuner_name(self, name):
        "Report tuner used"
        self.section(name)

    def section(self, name):
        cprint("-=[%s]=-" % name, 'magenta')

    def text(self, text):
        "print text value"
        print(text)

    def new_instance(self, instance, num_instances, remaining_budget):
        "Report the search of a new instance"
        msg = "New instance - Remaining Epoch Budget %s/%s Num Instances %s" % (
                remaining_budget, self.hypertuner.epoch_budget, num_instances)
        cprint(msg, 'yellow')
        self.setting("num params: %s" % instance.model_size)

    # TODO refactor to move the message out
    def done(self):
        msg = "Hypertuning complete - result in %s" % self.hypertuner.meta_data['server']['local_dir']
        cprint(msg, 'green')
 
    def setting(self, msg):
        color = SETTING_COLORS[0]
        cprint('|- %s' % msg, color)

    def info(self, msg):
        self.print_msg('INFO', msg, 'cyan')

    def error(self, msg):
        self.print_msg('ERROR', msg, 'red')

    def warning(self, msg):
        self.print_msg('Warning', msg, 'yellow')
    
    def highlight(self, msg):
        self.print_msg('', msg, 'green')

    def print_msg(self, lvl, msg, color):
        s = "[%s] %s" % (lvl, msg)
        cprint(s, color)
