import os
import sys

"""
Usage:
python convert_audio.py <train dir in> <test dir in> <train dir out> <test dir out> <bitrate>

This is for use converting folders of mp3 files, already seperated into train and test datasets, into wav files of a specified bitrate at a specified location,
for use as input into any of the train_model programs, especially for train_model_generator.py. 

Potential issues: only works orks for files without special characters in the title.
Make sure to use relative paths to data from this script
If you need to do this manually, you can use the command:

ffmpeg -i <in_path.mp3> -acodec pcm_s16le -ac 1 -ar <bitrate> <out_path.wav>

https://stackoverflow.com/questions/13358287/how-to-convert-any-mp3-file-to-wav-16khz-mono-16bit
https://janakiev.com/blog/python-shell-commands/

"""


# Converts a folder of mp3s to wavs at specified bitrate.
def convert(input, output):
    for subdir, dirs, files in os.walk(input):
        for file in files:
            # print os.path.join(subdir, file)
            if file.endswith(".mp3"):
                filepath = subdir + os.sep + file
                print(filepath)
                new_file = file.replace(".mp3", ".wav")
                output_path = output + os.sep + new_file
                command_string = "ffmpeg -i " + filepath + " -acodec pcm_s16le -ac 1 -ar " + str(bitrate) + " " + output_path
                os.system(command_string)


argv = sys.argv
input_dir_test = argv[1]
input_dir_train = argv[2]
output_dir_test = argv[3]
output_dir_train = argv[4]
bitrate = argv[5]

convert(input_dir_train, output_dir_train)
convert(input_dir_test, output_dir_test)