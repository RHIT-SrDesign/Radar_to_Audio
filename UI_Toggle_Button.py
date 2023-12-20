import PySimpleGUI as sg

"""
    Toggle Button Demo
    The background color of the button toggles between on and off
    Two versions are present... a simple button that changes text and a graphical one
    A HUGE thank you to the PySimpleGUI community memeber that donated his time and skill in creating the buttons!
    The text of the button toggles between Off and On
    
    Copyright 2021 PySimpleGUI
"""

def main():
    layout = [[sg.Text('A toggle button example')],
             [sg.Button('On', size=(3, 1), button_color='white on green', key='-B-'),  sg.Button('Exit')]]

    window = sg.Window('Toggle Button Example', layout)

    while True:             # Event Loop
        event, values = window.read()
        print(event, values)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        elif event == '-B-':                # if the normal button that changes color and text
            down = not down
            window['-B-'].update(text='On' if down else 'Off', button_color='white on green' if down else 'white on red')
        
    window.close()

