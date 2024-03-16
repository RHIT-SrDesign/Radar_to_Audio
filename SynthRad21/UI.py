import PySimpleGUI as sg
import matplotlib.pyplot as plt
from io import BytesIO
import UIHelper as ui
import Analysis 
from pygame import mixer
import pygame
import SDRDriver
# import SDRexample as SDR

sg.theme('DarkAmber')

mixer.init()
file_path = None
live = False
pause = False
running = False
is_playing = False
SDRLower = 0
SDRUpper = 1
numcaps = 1 # grab this signal 5 times
limplot = True # dont plot while capturing
control_col = [
    [sg.Text('Data Load Controls')],
    [sg.Button('Load Sim Data', key = '-Sim-'),sg.Button('Simulation Mode', key = '-Live-')],
    [sg.Button('Run Reciever', key = '-recieve-'),sg.Button('Process Recieved Data', key = '-Process-')],
    [sg.Frame('Volume',layout = [[sg.Slider(range = (0,100), orientation = 'h', key = '-Vol-')]])],
    [sg.Button('Play', key = '-Play-'),sg.Button('Pause', key = '-Pause-'),sg.Button('Stop', key = '-Stop-')],
    [sg.HorizontalSeparator()],
    [sg.Text('SDR Controls')],
    [sg.Text('Sweep Lower (MHz)'),sg.InputText(size=(20,1), key = '-Low-'), sg.Button('Submit', key = '-LF-')],
    [sg.Text('Sweep Upper (MHz)'),sg.InputText(size=(20,1), key = '-High-'), sg.Button('Submit', key = '-UF-')]
    ]

CFA  = [('\u2B24'+' No Freq Agility', 'red'), ('\u2B24'+' Frequency Agile', 'green')]
CFState=0
PRIA =[('\u2B24'+' No PRI Agility', 'red'), ('\u2B24'+' PRI Agile', 'green')]
PRIState=0
PWA =[('\u2B24'+' No PW Agility', 'red'), ('\u2B24'+' PW Agile', 'green')]
PWState = 0

image_col = [
    [sg.Text(text=CFA[CFState][0], text_color=CFA[CFState][1], key='INDICATOR1'), sg.Text(text=PRIA[PRIState][0], text_color=CFA[PRIState][1], key='INDICATOR2'),sg.Text(text=PWA[PWState][0], text_color=CFA[PWState][1], key='INDICATOR3')],
    [sg.HorizontalSeparator()],
    [sg.pin(sg.Image(key='-IMAGE-'))]

    
]


layout = [[sg.Column(control_col,element_justification='center',vertical_alignment='t'),sg.VSeparator(),sg.Column(image_col)]]
 
window = sg.Window('Radar Vis', layout, finalize=True, resizable=True)

pygame.init()

while True:
    event, values = window.read(timeout = 50)
    if event == sg.WIN_CLOSED:
        break
    if event == '-Sim-':
        if not live:
            plt.clf()
            file_path=sg.popup_get_file('Open',no_window = True)
            window['-IMAGE-'].update(visible=True)
            audio = mixer.Sound("audio.wav")
            if (file_path != None):
                image = Analysis.execute(file_path)
                print("image created")
                ui.draw_figure(window['-IMAGE-'], image)
                print("plot on screen now!")
    if event == '-Live-':
        live = not live
        window.Element('-Live-').update(text='Reciever Mode' if live else 'Simulation Mode', button_color='white on green' if live else 'black on gold')
        if live:
            print("running sweeps")
            while not SDRDriver.runProc(SDRLower,SDRUpper,numcaps,limplot):
                pass
    if event == '-Play-':
        is_playing = True
        audio_channel = mixer.Channel(0)
        audio_channel.play(audio)
    if is_playing == True:
        audio_channel.set_volume(values['-Vol-']/500)
        if not audio_channel.get_busy():
            audio_channel.queue(audio)
    if event == '-Pause-':
        window.Element('-Pause-').update(text='Pause' if pause else 'Resume')
        pause = not pause
        if pause:
            audio_channel.pause()
        else:
            audio_channel.unpause()
                
    if event == '-Stop-':
        is_playing = False
        window.Element('-Pause-').update(text='Pause')
        audio_channel.stop()

    if event == '-LF-':
        SDRLower = values['-Low-']
        print(SDRLower)
    
    if event == '-UF-':
        SDRUpper = values['-High-']
        print(SDRUpper)

    if event == '-Process-':
        [data, time, fft, freqs] = SDRDriver.returnData()
        
        fig2, (ax) = plt.subplots(nrows=1)
        ax.plot(freqs, fft.T)
        #plt.ylim(top=100,bottom=30)
        ax.set_title('Power Spectral Density')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power/Frequency (dB/Hz)')
        ax.set_gid(True)
        plt.close('all')
        image = fig2
        print("image created")
        ui.draw_figure(window['-IMAGE-'], image)
        print("plot on screen now!")

    

    
    # if event == 'Run Receiever':
    #     if running == False:
    #         running = SDR.SDRRun(running)
    #     else:
    #         pass
    

    # if event == '-Load-':
        

    if event == sg.WIN_CLOSED:
        break


window.close()