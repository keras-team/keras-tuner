"display utilities"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from terminaltables import SingleTable
from tabulate import tabulate
from colorama import init, Fore, Back, Style
init()  # colorama init

# Check if we are in ipython/colab
try:
    get_ipython().__class__.__name__
    from IPython.display import HTML
    ipython = True
except NameError:
    ipython = False
    display = print

FG = 0
BG = 1

# TODO: create a set of HTML color to allows richer display in colab 
colors = {
    'black': [Fore.BLACK, Back.BLACK],
    'red': [Fore.RED, Back.RED],
    'green': [Fore.GREEN, Back.GREEN],
    'yellow': [Fore.YELLOW, Back.YELLOW],
    'blue': [Fore.BLUE, Back.BLUE],
    'magenta': [Fore.MAGENTA, Back.MAGENTA],
    'cyan': [Fore.CYAN, Back.CYAN],
    'white': [Fore.WHITE, Back.WHITE],
}

styles = {
    "dim": Style.DIM,
    "normal": Style.NORMAL,
    "bright": Style.BRIGHT,
    "reset": Style.RESET_ALL
}

# shorthand functions
def section(text):
    if ipython:
        section = '<h1 style="font-size:18px">' + text + '</h1>'
        cprint(section, '#4527A0')
    else:
        section = '[' + text + ']'
        cprint(section, 'yellow')


def subsection(text):
    if ipython:
        section = '<h2 style="font-size:16px">' + text + '</h2>'
        cprint(section, '#7E57C2')
    else:
        section = '> ' + text + ''
        cprint(section, 'magenta', brightness='dim')


def setting(text, ident=0, idx=0, display=True):
    """ print setting
    
    Args:
        text (str): setting key:value as string
        ident (int, optional): Defaults to 0. Space indentation
        idx (int, optional): Defaults to 0. index of setting to rotate color.
        display (bool, optional): Defaults to True. Display or return settings
    
    Returns:
        str: setting value if display=False, None otherwise 
    """

    s = ' ' * ident
    s += '|-' + text
    if idx % 2:
        color = 'cyan'
    else:
        color = 'blue'

    if display:
        cprint(s, color)
    else:
        return colorize(s + '\n', color)

def highlight(text):
    if ipython:
        text = '<span style="font-size:14px">' + text + '</span>'
        cprint(text, '#64DD17')
    else:
        cprint(text, 'green')

# Charts

def print_bar_chart(val, max_val, title=None, left='', right='', 
                    color='green', length=80):

    bar = make_bar_chart(val, max_val, title=title, left=left, right=right,
                         color=color, length=length)
    display(bar)


def make_bar_chart(val, max_val, title=None, left='', right='', 
                   color='green', length=80):
    full_block = '█'
    empty_block = '░'
    half_block = '▒'

    # building the bar
    bar = ''
    num_full = length * val/float(max_val)
    bar += full_block * int(num_full)
    if not (num_full).is_integer():
        bar += half_block
    bar += empty_block * (length - len(bar))
    
    # colorize
    bar = colorize(bar, color)

    # adding left/right text if needed
    row = []
    if left:
        row.append(left)
    row.append(bar)
    if right:
        row.append(right)

    st = SingleTable([row], title)
    st.inner_column_border = False
    return st.table

# Low level function

def cprint(text, color, bg_color=None, brightness='normal'):
    """ Print given piece of text with color
    Args:
        text (str): text to colorize
        color (str): forground color
        bg_color (str, optional): Defaults to None. background color.
        brightness (str, optional): Defaults to normal. Text brightness.
    """

    text = colorize(text, color, bg_color, brightness)

    # HTMLify if needed
    if ipython and isinstance(text, str):
        text = HTML(text)
    display(text)


def colorize(text, color, bg_color=None, brightness='normal'):
    """ Colorize a given piece of text
    Args:
        text (str): text to colorize
        color (str): forground color
        bg_color (str, optional): Defaults to None. background color.
        brightness (str, optional): Defaults to normal. Text brightness.

    Returns:
        str: colorized text
    """

    if color not in colors and not ipython:
        msg = "Foreground color invalid:%s" % color
        raise ValueError(msg)

    if bg_color and bg_color not in colors and not ipython:
        "Background color invalid:%s" % bg_color
        raise ValueError(msg)

    if brightness not in brightness and not ipython:
        raise ValueError("Brightness invalid:" + brightness)

    text = str(text)  # in case user pass a float/int

    # foreground color
    if ipython:
        text = text.replace('\n', '<br>')
        h = '<span style="color:%s">' % color
        text = h + text
    else:
        text = colors[color][FG] + text
    # background if needed
    if bg_color and not ipython:
        text = colors[bg_color][BG] + text

    # brightness if neeed
    if brightness != 'normal' and not ipython:
        text = styles[brightness] + text

    # reset
    if ipython:
        text = text + '</span>'
    else:
        text = text + styles['reset']

    return text


# TABLE 
def print_table(rows, title=None):
    """ Print data as a nicely formated ascii table
    Args:
        rows (list(list)): data to display as list of lists.
        title (str, optional): Defaults to None. Table title
    """
    display(make_table(rows, title))


def make_table(rows, title=None):
    """ Format list as a pretty ascii table
    Args:
        rows (list(list)): data to display as list of lists.
        title (str, optional): Defaults to None. Table title
    Returns:
        str: string representing table
    """
    if ipython:
        headers = rows[0]
        body = rows[1:]
        table = tabulate(body, headers, tablefmt="html")
        table = HTML(table)
    else:
        st = SingleTable(rows, title)
        table = st.table
    return table


def make_combined_table(array_rows):
    """ Build a table of tables

    Args:
        array_rows (list(list)): Array of tables rows to combine
    Returns:
        str: string representing table
    """

    if ipython:
        # compute the size for each col
        col_size = str(int(100 / len(array_rows)) - 5) + '%'
        gtc = [col_size] * len(array_rows)
        table = """
        <style>
            .wrapper {
                display: grid;
                grid-template-columns: %s;
                grid-gap: 10px;
            }
        </style>
        <div  class="wrapper">
        """ % (" ".join(gtc))
        for rows in array_rows:
            table += '<div>'
            headers = rows[0]
            body = rows[1:]
            table += tabulate(body, headers, tablefmt="html")
            table += '</div>'
        table += "</div>"
        return HTML(table)
    else:
        tables = []
        for rows in array_rows:
            tables.append(make_table(rows))
        combined_table = SingleTable([tables])
        combined_table.outer_border = False
        combined_table.inner_column_border = False
        return combined_table.table


def print_combined_table(array_rows):
    """ Build a table of tables and print it

    Args:
        array_rows (list(list)): Array of tables rows to combine
    """
    table = make_combined_table(array_rows)
    display(table)