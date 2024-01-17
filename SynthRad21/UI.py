import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import BytesIO

sg.theme('DarkAmber')


file_path = None
 
control_col = [
    [sg.Button('Load Sim Data', key = '-Sim-'),sg.Button('Connect Live Data', key = '-Live-')],
    [sg.Frame('Volume',layout = [[sg.Slider(range = (0,100), orientation = 'h', key = '-BLUR-')]])],
    [sg.Button('Pause', key = '-Pause-'),sg.Button('Play', key = '-Play-')]
    ]

table_content = []

CFA  = [('\u2B24'+' No Freq Agility', 'red'), ('\u2B24'+' Frequency Agile', 'green')]

CFState=0
PRIA =[('\u2B24'+' No PRI Agility', 'red'), ('\u2B24'+' PRI Agile', 'green')]
PRIState=0
PWA =[('\u2B24'+' No PW Agility', 'red'), ('\u2B24'+' PW Agile', 'green')]
PWState = 0

image_col = [
    [sg.Text(text=CFA[CFState][0], text_color=CFA[CFState][1], key='INDICATOR1'), sg.Text(text=PRIA[PRIState][0], text_color=CFA[PRIState][1], key='INDICATOR2'),sg.Text(text=PWA[PWState][0], text_color=CFA[PWState][1], key='INDICATOR3')],
    [sg.Image('SynthRad21\Angry_bear.GIF', key = 'display')],

    
]

control_col2 = [
            [sg.Text('Bandwidth (MHz)')],
            [sg.InputText()],
            [sg.Text('Center Frequency (MHz)')],
            [sg.InputText()]
    ]
layout = [[sg.Column(control_col,element_justification='center',vertical_alignment='t'),sg.VSeparator(),sg.Column(image_col),sg.VSeparator(),sg.Column(control_col2,vertical_alignment='t')]]
 
window = sg.Window('Radar Vis', layout, finalize=True, resizable=True)

 
while True:
    event, values = window.read(timeout = 50)
    if event == sg.WIN_CLOSED:
        break
    if event == '-Sim-':
        file_path=sg.popup_get_file('Open',no_window = True)
        window.Element('display').update(filename=file_path)
    if event == '-Live-':
        window.Element('-Live-').Update(('Not Availible'), button_color=(('white', ('red'))))
window.close()