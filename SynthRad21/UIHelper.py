import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.pyplot as plt
import io
import time


def draw_figure(element, figure):
    """
    Draws the previously created "figure" in the supplied Image Element

    :param element: an Image Element
    :param figure: a Matplotlib figure
    :return: The figure canvas
    """
    print("drawing fig")
    #plt.close('all')  # erases previously drawn plots
    canv = FigureCanvasAgg(figure)
    buf = io.BytesIO()
    canv.print_figure(buf, format='png')
    if buf is not None:
        buf.seek(0)
        element.update(data=buf.read())
        print("done printing buffer got  stuff")
        return canv
    else:
        print("done printing empty buffer")
        return None
    
