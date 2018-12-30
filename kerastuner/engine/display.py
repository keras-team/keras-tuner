"display utilities"

from colorama import init, Fore, Back, Style
from terminaltables import SingleTable

#colorama init
init()

FG = 0
BG = 1

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

def cprint(text, color, bg_color=None, brightness='normal'):
    """ Print given piece of text with color
    Args:
        text (str): text to colorize
        color (str): forground color
        bg_color (str, optional): Defaults to None. background color.
        brightness (str, optional): Defaults to normal. Text brightness.
    """
    print(colorize(text, color, bg_color, brightness))


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

    if color not in colors:
        raise ValueError("Forground color invalid:" + color)

    if bg_color and bg_color not in colors:
        raise ValueError("Backgroun color invalid:" + bg_color)

    if brightness not in brightness:
        raise ValueError("Brightness invalid:" + brightness)

    # foreground color
    text = colors[color][FG] + str(text)

    # background if needed
    if bg_color:
        text = colors[bg_color][BG] + text

    # brightness if neeed
    if brightness != 'normal':
        text = styles[brightness] + text

    # reset
    text = text + styles['reset']

    return text


# TABLE 


def print_table(rows, title=None):
    """ Print data as a nicely formated ascii table
    Args:
        rows (list(list)): data to display as list of lists.
        title (str, optional): Defaults to None. Table title
    """
    print(get_table(rows, title))


def get_table(rows, title=None):
    """ get data as a nicely formated ascii table
    Args:
        rows (list(list)): data to display as list of lists.
        title (str, optional): Defaults to None. Table title
    Returns:
        str: string representing table
    """
    table = SingleTable(rows, title)
    return table.table


def get_combined_table(array_rows):
    """ Build a table of tables

    Args:
        array_rows (list(list)): Array of tables rows to combine
    Returns:
        str: string representing table
    """

    tables = []
    for rows in array_rows:
        tables.append(get_table(rows))
    combined_table = SingleTable([tables])
    combined_table.outer_border = False
    combined_table.inner_column_border = False
    return combined_table.table


def print_combined_table(array_rows):
    """ Build a table of tables and print it

    Args:
        array_rows (list(list)): Array of tables rows to combine
    """
    table = get_combined_table(array_rows)
    print(table)
