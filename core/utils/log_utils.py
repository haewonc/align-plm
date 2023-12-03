import termcolor
import sys

def toRed(content):
    return termcolor.colored(content,"red",attrs=["bold"])

def toGreen(content):
    return termcolor.colored(content,"green",attrs=["bold"])

def toBlue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])

def toCyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])

def toYellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])

def toMagenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])

def toGrey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])

def toWhite(content):
    return termcolor.colored(content,"white",attrs=["bold"])
    
def print_progress(phase, epoch, max_epoch, iter, max_iter, elapsed_time, loss_names, loss_vals, metric_names=[], metric_vals=[]):
    CURSOR_UP_ONE = '\x1b[1A' 
    ERASE_LINE = '\x1b[2K'

    sys.stdout.write('{}[{}] {} {} {} [{}] '.format(
        CURSOR_UP_ONE + ERASE_LINE,
        toWhite(phase),
        toWhite('{} '.format('EP')) + toCyan('{}/{}'.format(epoch, max_epoch)),
        toWhite('{} '.format('ITER')) + toCyan('{}/{}'.format(iter+1, max_iter)),
        toWhite('ETA') + toGreen(' {:5.2f}s'.format(elapsed_time / (iter+1) * (max_iter-iter))),
        toWhite('LOSS')
        )
    )
    for name, val in zip(loss_names, loss_vals):
        sys.stdout.write('{} {:.3f} '.format(toWhite(name), val))
    
    if len(metric_names) > 0:
        sys.stdout.write('['+toWhite('METRIC')+'] ')

        for name, val in zip(metric_names, metric_vals):
            sys.stdout.write('{} {:.3f} '.format(toWhite(name), val))
        
    sys.stdout.write('\n')

def print_total(phase, epoch, max_epoch, loss_names, loss_vals, metric_names=[], metric_vals=[]):
    CURSOR_UP_ONE = '\x1b[1A' 
    ERASE_LINE = '\x1b[2K'

    sys.stdout.write('{}[{}] {} [{}] '.format(
        CURSOR_UP_ONE + ERASE_LINE,
        toWhite(phase + " TOTAL"),
        toWhite('{} '.format('EP')) + toCyan('{}/{}'.format(epoch, max_epoch)),
        toWhite('AVG LOSS')
        )
    )
    for name, val in zip(loss_names, loss_vals):
        sys.stdout.write('{} {:.2f} '.format(toWhite(name), val))

    if len(metric_names) > 0:
        sys.stdout.write('['+toWhite('AVG METRIC')+'] ')

        for name, val in zip(metric_names, metric_vals):
            sys.stdout.write('{} {:.2f} '.format(toWhite(name), val))
        
    sys.stdout.write('\n\n')