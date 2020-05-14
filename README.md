# musicrgan
As the name for this repo suggests, this was intended to be a recurrennt generative adversarial network that created music. As a quick exploration of this repo will also show, that is not what happened. As a result, this repo contains the code for three attempts at generating music, each with varying levels of success:

1. A Recurrent GAN
2. A Regressive model (my design)
3. A Regressive model (design from [here](https://github.com/unnati-xyz/music-generation))


## Background
This project started as a broad question: could I take what I had learned about neural networks and use it to create something to resemble music?

The answer is "sort of."

I wanted to build a network that needed as little feature engineering as possible, something that could take an input of audio files and generate something similar, ideally, a generative adversarial network.

Essentially, audio files are sequences of data, so the foundational techniques applied to sequence data seemed to be a good place to start. However, audio files have two key differences from most types of sequence data often fed into neural networks. First, there is much more data. A standard two minute audio file encoded with a 44.1k bitrate has over 5,000,000 samples! That is huge. Second, that data is much less information dense than most other types of data. Maybe the song has three or four notes per second, but that three to four notes is spread out over 44,100 data samples! This creates some unique challenges in dealing with this data. I certainly excpected that the network would take far longer to train.

<Insert visualization of audio data>

### Environment
Hardware:
- 1x Nvidia Titan RTX graphics card with 24gb of GDDR6 memory. The cluster had 4, but I was allocated one for the duration of the project. 

### Building a Music GAN
My first step was to design a GAN that used LSTMs (Long-Short Term Memory) layers in both the generator and discrimator components of the GAN. My plan was to get the network working, then scale it up to the problem set. However, I ran into some issues here with Tensorflow itself. Tensorflow's train_on_batch function, which was essential to how I had set up the GAN, gave an error when I attempted to run it with the Titan RTX. After a few hours debugging, I switched approaches from build a GAN, then make it work on music, to make a simple music model, then make it into a GAN. I never made it far enough to come back to the GAN concept.

<Insert model summary>

### A basic regressive music generator
This model attempts to take an audio file and create a model that generates a solution one sample at a time. Ultimately, it fails to create anything remotely resembling music, for several reasons. I loosely based this model of of this article: https://medium.com/intel-student-ambassadors/music-generation-using-lstms-in-keras-9ded32835a8f
1. The design of the model, which relies on LSTM generators, works best on shorter sequences. However, short sequences do not contain nearly enough information to allow the network to create anything resembling music
2. As I learned later, the models were simply not large enough to learn to make music.

Sample results are in regressive_1/results. The file recurrent4.wav is the best demo of the network, though even it just creates a high-pitched whine. 
<Insert model summary, training results>


#### Execution
You can train these models using the files in the folder regressive_1.

The models I generated are stored in the folder "models"
You can generate new audio files using make_music.py. 

Usage as follows:` python make_music <model_path> <input_window> <bitrate> <num songs> <savepath> `
model_path is the relative path to the model you want to use to make the song (must have been generated with musicrganmodels2.py, which also resides in regressive_1).
input_window is how long a start sequence you want to feed into the model
bitrate is how many samples to generate per second
num_songs is how many songs to generate
savepath is where to save them.

You will need to make one additional change to the file itself. Line 48 contains the path to the seed data. Replace that with a path to the data you want to be used to seed the audio.


### Another Regressive Model
At this point, I shrunk my goal again and set a goal of merely implementing anything that generated realistic music. I found code from Padmaja Bhagwat, an Indian developer, that uses a regressive model for generating audio. 

Their design model worked by chunking the data, instead of trying to make predictions sample by sample, then used an LSTM to predict the next chunk of data.

Another key feature of their design was using a Fast Fourier Transform(FFT) to break the audio into its component frequencies. A FFT transforms signal data from the time domain to the frequency domain. Hence, this is why the data must be chunked, to preserve that temporality. I reused most of the helper functions from this code, defined in helpers.py

It takes an input of a ten block sequence, and then output another ten block sequence, that included the last 9 blocks of input plus the predicted next block. 

<Insert model summary>

The resulting model had nearly 200 million parameters. I verified the general concept of the model by intentionally overtraining it on 6 YoYo Ma cello songs, the result are in the folder regressive2/results.
model1_song2.wav is the result of training the network for 1000epochs.
model1_song3.wav is the result of training that model for an additional 500 epochs, for a total of 1500 epochs. 
As you can see, these songs sound very remincent of the training audio (also availible in github in the folder yoyoma_dataset).


I also attempted to modify the network slightly, so that it would generate only the next block instead of the next block, plus the last 9. The code for this attempt is in train_model2.py. The code runs, but the model is not successful. You can hear the result in results/model2_1.wav

Finally, I attempted to train the model on a far larger dataset, to see how it would perform. For this, I used the Free Music Archive Small dataset, which consists of 10k 30 second song segments. You can obtain it [here](https://github.com/mdeff/fma)

After training the network for nearly 4 days, the result is still barely recognizable as music. Recording how well this model is actually doing at converging is difficult due to a design flaw in my code, which trains the model in batches of 10 epochs, creating a new history object every 10 epochs that overwrites the old one, erasing valuable training information.s I didn't realize this until the training had already occured for several days. So far, this model also simple produces what almost sounds like crowd noises, but is not music. This can be found in regressive2/results/model3_2.wav


#### Execution
To replicate the the first and second odels in regressive2, follow these steps:

1. Use the command `ffmpeg -i <in_path.mp3> -acodec pcm_s16le -ac 1 -ar 44100 <out_path.wav>` to convert the files in yoyoma_dataset into wav format
2. Update the train_path and test_path variables in train_model.py with the locations of the coverted wavs, in test/train split
3. Use train_model.py (or train_model2.py)to train your model and, if you wish, generate audio. Usage as follows:
	train_model.py <mode> <modelpath> <epochs> <songpath>

	- Mode
	"new" creates new model at model path
	"continue" loads old model, trains
	"generate" only generates a song from the given model. Still requires all four args
	- modelpath - where the model will be, or is, saved
	- how many epochs to train model
	- path to place the exported song, should end in filename.wav.

To use train_model_generator, which works off a far larger dataset, follow these steps:
1. Use convert_audio.py to convert folders of train and test data in mp3 format into wav format at a specified bitrate (I used 16000)
`python convert_audio.py <train dir in> <test dir in> <train dir out> <test dir out> <bitrate>`

2. Update train_model_generator to include the new data paths and correct bitrate. Execute using hte same commands as train_model.py.


### Conclusion
By the end of this adventure, I realized a few things about my earlier approaches.
1. You need very large networks to do anything with audio data. Both the GAN and the first regressive model I built were far to small to have any chance of actually learning the complex patterns needed to generate music.

2. I approached the problem in the wrong order. Rather than going from hard (GAN) to easy (Copying a regressive model), I should have started the other way around. This misconstrued approach was mostly due to my shallow understanding of sequence generation with neural networks for the first few months of the semester. Ironically, now that the semester is ended and the project is done, I feel as if I have enough knowledge to actually attempt to start the project anew.







