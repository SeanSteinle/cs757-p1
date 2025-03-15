# Assignment 1 Report
*Kiya Aminfar, Sean Steinle*

## The Simple VAE Model

### A Key Distinction: Raw Audio vs. Spectrograms

We started off by creating a simple VAE with only a few layers for the encoder and decoder. We also explored the audio modality since neither of us had much experience. From working with the data, we found that there is a key distinction to be made in whether audio is represented as raw audio in a .wav file or whether the audio has been transformed into a spectrogram. You can see this difference qualitatively in our `notebooks/simple_vae.ipynb` notebook where we play the raw audio, a version of the raw audio which was translated into a spectrogram and back, and our predicted spectrogram translated into audio. This distinction is key, because while our first samples were very poor (essentially white noise), this gap in quality between raw audio and transformed audio will act as a ceiling on our model's performance since we are generating spectrogram samples.

### Performance and Avenues for Improvement

As mentioned above, our performance was fairly poor on our first run. This is likely due to a few factors: we only trained for 20 epochs, we only used 1000 samples to train, and our model was overly simplistic. Luckily, these limitations also act as avenues to improve our model!

In case you don't want to run the notebook, [here](outputs/samples/simple_vae.wav) is a sample from our first run.

***Editorial Note:*** Looking back, this occurred because we were not doing sufficient audio processing! We were not padding and truncating our audio files and also we did not 'unnormalize' our generated spectrograms before converting them to audio. This resulted in an absolutely wretched 4 second clip for each of our predictions.

### An Aside: Saving Models

Something I've accidentally spent a lot of time on is trying to save out our VAE model. This would be convenient so that we do not need to retrain our model from scratch each time we create it. However, saving Keras objects is non-trivial, and must be done carefully. Something I've quickly learned is that it's probably best to keep your model very modular as opposed to monolithic. For example, it would be better to almost treat the encoder and decoder as separate models which can be passed in to assemble the VAE object, rather than trying to serialize the entire VAE object.

Here are some threads I've been looking at for reference:
1. [Blog on how to build a VAE in Keras](https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/)
2. [Reddit thread on saving Keras VAE's](https://www.reddit.com/r/learnmachinelearning/comments/t4dbmb/how_to_save_vae_model_made_by_keras/)

## An Improved VAE Model

### Revisiting the Spectrogram - Audio Dichotomy

The challenges listed above led us to develop a new model in `notebooks/improved_vae.ipynb`, inspired by a [Medium blogpost](https://yuehan-z.medium.com/how-to-train-your-ml-models-music-generation-try-it-out-d4c0ab01c9f4). This model is more modular but more importantly we learned how to handle audio data more adeptly. This is evident in our best sample from this model. I've linked the audio from the [ground-truth spectrogram](outputs/samples/improved_vae_gt.wav) and our [generation](outputs/samples/improved_vae_pred.wav)--can you guess which is which?