import speech_recognition as sr
import tkinter as tk

is_espeak_installed = False

try:
    from espeak import espeak 
    is_espeak_installed = True
except:
    from os import system


# def get_voice_input():

#     r = sr.Recognizer()
#     mic = sr.Microphone()
#     # sr.Microphone.list_microphone_names()
#     mic = sr.Microphone(device_index=0)
#     while True:

#         root = tk.Tk()
#         result = None

#         def cont():
#             root.destroy()

#         def again():
#             import pdb; pdb.set_trace()
#             with mic as source:
#                 if is_espeak_installed:
#                     espeak.synth("What would you like me to do?")
#                 else:
#                     system("say What would you like me to do?")
#                 r.adjust_for_ambient_noise(source)
#                 audio = r.listen(source)
#                 result = r.recognize_google(audio)

#                 if is_espeak_installed:
#                     espeak.synth(result)
#                 else:
#                     system("say " + result)

#         text_widg = tk.Text(root, height=2, width=30)
#         cont_widg = tk.Button(root, text ="Continue", command = cont)
#         again_widg = tk.Button(root, text ="Again", command = again)
#         text_widg.pack()
#         cont_widg.pack()
#         again_widg.pack()
#         text_widg.insert(tk.END, "Just a text Widget\nin two lines\n")
#         tk.mainloop()

#         return result


def get_voice_input():

    while True:
        type_or_speak = None
        type_or_speak = input("Type(1) or speak(2) command. Or stop(3). ")

        if type_or_speak == "1":
            return input("Type command. ")
        elif type_or_speak == "2":
            r = sr.Recognizer()
            mic = sr.Microphone()
            # sr.Microphone.list_microphone_names()
            mic = sr.Microphone(device_index=0)
            while True:
                with mic as source:
                    print("Start speaking")
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source)
                    try:
                        result = r.recognize_google(audio)
                        print(result)
                    except:
                        print("Please repeat yourelf.")
                    get_correct_res = None
                    while get_correct_res == None:
                        get_correct_res = input(
                            "Is this what you want (y/n)? ")
                        if get_correct_res == "y":
                            return result
        elif type_or_speak == "3":
            return "stop" 