
def progress_bar(current, total, bar_length=20, name='Progress'):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces  = ' ' * (bar_length - len(arrow))

    print('%s: [%s%s] %d %%' % (name, arrow, spaces, percent), end='\r')
