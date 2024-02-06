import PySimpleGUI as sg

sg.theme("DarkBlue3")
sg.set_options(font=("Courier New", 16))

status = [('\u2B24'+' Disconnect', 'red'), ('\u2B24 '+' Connect', 'green')]
state = 0

layout = [
    [sg.Text(text=status[state][0], text_color=status[state][1], size=(20, 1),
        justification='center', font=("Courier New", 24), key='INDICATOR')],
    [sg.Column([[sg.Button('Connect'), sg.Button('Disconnect')]], justification='center')],
]

window = sg.Window('Title', layout, finalize=True)

while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, "Exit"):
        break
    elif event == 'Connect':
        state = 1
    elif event == 'Disconnect':
        state = 0
    if event in ('Disconnect', 'Connect'):
        value, text_color = status[state]
        window['INDICATOR'].update(value=value, text_color=text_color)

window.close()