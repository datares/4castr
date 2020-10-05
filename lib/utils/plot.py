import numpy as np
import matplotlib.pyplot as plt

# plt.show()


def start_plot(p_vals, open_prices):
    """TODO"""
    x = range(len(p_vals))
    y1 = p_vals
    y2 = open_prices

    # You probably won't need this if you're embedding things in a tkinter plot...
    plt.ion()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(112)
    line1, _ = ax1.plot(x, y1, 'r-')
    line2, _ = ax1.plot(x, y2, 'r-') 
    

def update_plot(fig, line1, line2, p_vals, open_prices):
    """TODO"""
    fig.canvas.flush_events()
    line1.set_ydata(p_vals)
    line2.set_ydata(open_prices)
    fig.canvas.draw()
