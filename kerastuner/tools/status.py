#import asciichartpy
import sys
import os
import argparse
import math
import numpy as np
import json
import platform
from kerastuner.abstractions.display import make_bar_chart, make_combined_table
from kerastuner.abstractions.display import colorize, setting, cprint, make_table
import time
from time import gmtime, strftime
from etaprogress.components.eta_conversions import eta_letters

from kerastuner.abstractions.system import System


def parse_args():
    parser = argparse.ArgumentParser(description='Kerastuner status monitor')
    parser.add_argument(
        '--input_dir',
        '-i',
        type=str,
        default='results/',
        help='Directory containing tuner results')
    parser.add_argument(
        '--graphs',
        '-g',
        type=str,
        default='loss',
        help='Comma separated list of key metrics to graph.')
    parser.add_argument(
        '--refresh_rate',
        '-r',
        type=int,
        default=2,
        help='Refresh rate in second')
    parser.add_argument('--debug', '-d', type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        cprint("[Error] Invalid Input directory %s" % args.input_dir, 'red')
        parser.print_help()
        quit()
    return args


def read_status(dir):
    fname = os.path.join(dir, 'status.json')
    status = json.loads(open(fname).read())
    return status


def bar(total, done, eta, title, color):
    lside = 'Epochs %s/%s' % (done, total)
    rside = 'ETA:%s' % eta_letters(eta, shortest=False)

    return make_bar_chart(
        done,
        total,
        color=color,
        title=title,
        left=lside.ljust(15),
        right=rside.rjust(16))


def clear():
    if platform.system() == 'Windows':
        os.system('cls')  # on windows
    else:
        os.system('clear')  # on linux / os x


LAST_EPOCH_COUNT = -1


def display_status(status, system):
    global LAST_EPOCH_COUNT
    # FIXME better ASCII art
    display = colorize('kerastuner status', 'magenta')
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

    if done != LAST_EPOCH_COUNT:
        refresh = True
        LAST_EPOCH_COUNT = done
    else:
        refresh = False

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
        fields = [
            'trained_models', 'collisions', 'invalid_models',
            'over_size_models'
        ]
        for k in fields:
            stats.append([k.replace('_', ' '), md[k]])

            info = [['info', ' '], ['project', status['project']],
                    ['architecture', status['architecture']],
                    ['tuner', status['tuner']['name']],
                    ['Num GPU', status['server']['num_gpu']]]

        system_info = system.get_status()
        gpus = [['GPU', 'Usage', 'Mem', 'Temp']]
        for g in system_info['gpu']:
            idx = g["index"]
            name = g["name"]
            usage = "%s%%" % g["usage"]
            mem = "%s/%sM" % (g["memory"]['used'], g["memory"]["total"])
            temp = "%s%s" % (g["temperature"]["value"],
                             g["temperature"]["unit"])
            gpus.append(["%s : %s" % (name, idx), usage, mem, temp])
    display += make_combined_table([stats, metrics, gpus]) + "\n"

    # for metric in sorted(status['epoch_metrics'].keys()):

    #     display += make_table([["% 40s% 40s" % (metric, "")]]) + "\n"
    #     epoch_metrics = status['epoch_metrics'][metric]

    #     if len(epoch_metrics) > 32:

    #         epoch_metrics = epoch_metrics[-32:]
    #     display += asciichartpy.plot(epoch_metrics, cfg={
    #         "minimum": min(epoch_metrics),
    #         "maximum": max(epoch_metrics),
    #         "height": 8,
    #     }) + "\n"

    # update the display at once
    if refresh:
        clear()
        print(display)


def status(debug=0):
    args = parse_args()
    system = System()  # system monitoring
    while 1:
        try:
            status = read_status(args.input_dir)
        except:
            if args.debug:
                import traceback
                traceback.print_exc()
                quit()
            time.sleep(1)

        display_status(status, system)
        time.sleep(args.refresh_rate)


if __name__ == '__main__':
    status()
