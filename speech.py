import speech_recognition as sr


def extract_audio(audio, r, input_type=None):
    try:
        result = r.recognize_google(audio)
        print(result)
        get_correct_res = None
        if input_type == "speak":
            while get_correct_res == None:
                get_correct_res = input(
                    "Is this what you want (y/n)? ")
                if get_correct_res == "y":
                    return result
        else:
            return result
    except:
        print("Please repeat yourelf or upload different file.")
        return None


def get_voice_input():

    while True:
        type_or_speak = input(
            "Type(1), speak(2), or voice file(3) command. Stop(4): ")
        if type_or_speak == "1":
            return input("Type command: ")
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
                    text = extract_audio(audio, r, "speak")
                    if text is not None:
                        return text
        elif type_or_speak == "3":
            r = sr.Recognizer()
            print("Must be: PCM WAV, AIFF/AIFF-C, or Native FLAC")
            audio_file_name = input(
                "Enter audio file name including extension: ")
            audio_file = sr.AudioFile(f"./audio_commands/{audio_file_name}")
            with audio_file as source:
                audio = r.record(source)
                return extract_audio(audio, r)
        elif type_or_speak == "4":
            return "stop"
