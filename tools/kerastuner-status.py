import sys
import os
import argparse
import json
import platform
from pathlib import Path
from kerastuner.engine.display import make_bar_chart, make_combined_table
from kerastuner.engine.display import colorize, setting, cprint, make_table
import time
from time import gmtime, strftime
from art import text2art
from etaprogress.components.eta_conversions import eta_letters

from kerastuner.utils import get_gpu_usage


def parse_args():
    parser = argparse.ArgumentParser(description='Kerastuner status monitor')
    parser.add_argument('--input_dir', '-i', type=str, default='results/',
                        help='Directory containing tuner results')
    parser.add_argument('--refresh_rate', '-r', type=int, default=2, 
                        help='Refresh rate in second')
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        cprint("[Error] Invalid Input directory %s" % args.input_dir, 'red')
        parser.print_help()
        quit()
    return args


def read_status(dir):
    fname = Path(dir) / 'status.json'
    status = json.loads(open(fname).read())
    return status


def bar(total, done, eta, title, color):
    lside = 'Epochs %s/%s' % (done, total)
    rside = 'ETA:%s' % eta_letters(eta, shortest=False)
    
    return make_bar_chart(done, total, color=color, 
                          title=title, left=lside.ljust(15), 
                          right=rside.rjust(16))

def clear():
    if platform.system() == 'Windows':
        os.system('cls') # on windows
    else:
        os.system('clear') # on linux / os x

def display_status(status):
    display = colorize(text2art('kerastuner status'), 'magenta')
    #display += colorize(art, 'magenta')
    display += '\n'
    # Tuner eta
    display += '\n'
    total = status['tuner']['epoch_budget']
    done = total - status['tuner']['remaining_budget']
    eta = status['tuner']['eta']
    display += bar(total, done, eta, 'Tuning progress', 'green')
    
    # Model eta
    display += '\n'
    total = status['tuner']['max_epochs']
    done = status['current_model']['epochs']
    eta = status['current_model']['eta']
    display += bar(total, done, eta, 'Current model progress', 'blue')

    display += '\n'    
    metrics = [['metric', 'best model', 'current model']] 
    for k in status['statistics']['best'].keys():
        best = status['statistics']['best'][k]
        if best == -1 or best == sys.maxsize:
            best = 'n/a'
        else:
            best = round(best, 4)
        current = status['statistics']['latest'][k]
        current = round(current, 4)
        # TODO fix callback to know which one are best based of direction
        if current == best:
            current = colorize(current, 'cyan')
        metrics.append([k, best, current])
    
        #tuner_data = [['Error', 'count']]
        md = status['tuner']
        stats = [['statistics', 'count']]
        fields = ['trained_models', 'collisions', 'invalid_models', 'over_size_models']
        for k in fields:
            stats.append([k.replace('_', ' '), md[k]])

        info = [
                ['info', ' '],
                ['project', status['project']],
                ['architecture', status['architecture']],
                ['tuner', status['tuner']['name']],
                ['Num GPU', status['server']['num_gpu']]
               ]

        smi = get_gpu_usage()
        gpus = [['GPU', 'Usages', 'Mem', 'Temp']]
        for g in smi:
            cpu = "%s%%" % g[1]
            mem = "%s/%sM" % (g[2].strip(), g[3].strip())
            temp = "%sC" % (g[5])
            gpus.append([g[0], cpu, mem, temp])
    display += make_combined_table([stats, metrics, gpus])

    # update the display at once
    print(display)

args = parse_args()

while 1:
    try:
        status = read_status(args.input_dir)
    except:
        clear()
        time.sleep(1)
    clear()
    display_status(status)
    time.sleep(args.refresh_rate)
