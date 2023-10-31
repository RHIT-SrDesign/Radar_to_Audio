import time
from matplotlib import pyplot as plt
import numpy as np

def live_update(bw,Ax1=False,Ax2=False,Ax3=False,blit = False,t=1.0):
    x = bw
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)


    line, = ax1.plot([], lw=1)
    text = ax1.text(0.8,0.5, "")

    ax1.set_xlim(x.min(), x.max())
    ax2.set_xlim()
    ax3.set_xlim()

    ax1.set_ylim([-1.1, 1.1])
    ax2.set_ylim()
    ax3.set_ylim()

    fig.canvas.draw()   # note that the first draw comes before setting data 


    if blit:
        # cache the background
        axbackground = fig.canvas.copy_from_bbox(ax1.bbox)
        ax1background = fig.canvas.copy_from_bbox(ax1.bbox)

    plt.show(block=False)


    t_start = time.time()
    k=0.

    for i in np.arange(1000):

        line.set_data(x, np.sin(x/3.+k))
        tx = ('Mean Frame Rate:\n {fps:.3f}FPS'.format(fps= ((i+1) / (time.time() - t_start)) ) )
        text.set_text(tx)
        #print tx
        k+=0.01


        if blit:
            # restore background
            fig.canvas.restore_region(axbackground)
            fig.canvas.restore_region(ax1background)

            # redraw just the points
            ax1.draw_artist(line)

            # fill in the axes rectangle
            fig.canvas.blit(ax1.bbox)

            # in this post http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
            # it is mentionned that blit causes strong memory leakage. 
            # however, I did not observe that.

        else:
            # redraw everything
            fig.canvas.draw()

        fig.canvas.flush_events()
        #alternatively you could use
        #plt.pause(0.000000000001) 
        # however plt.pause calls canvas.draw(), as can be read here:
        #http://bastibe.de/2013-05-30-speeding-up-matplotlib.html



x = np.linspace(0,50., num=100)
live_update(x,blit=True)   # 175 fps
#live_update_demo(False) # 28 fps