# Assignment 1 Report
*Kiya Aminfar, Sean Steinle*

## The Simple VAE Model

### A Key Distinction: Raw Audio vs. Spectrograms

We started off by creating a simple VAE with only a few layers for the encoder and decoder. We also explored the audio modality since neither of us had much experience. From working with the data, we found that there is a key distinction to be made in whether audio is represented as raw audio in a .wav file or whether the audio has been transformed into a spectrogram. You can see this difference qualitatively in our `simple_vae.ipynb` notebook where we play the raw audio, a version of the raw audio which was translated into a spectrogram and back, and our predicted spectrogram translated into audio. This distinction is key, because while our first samples were very poor (essentially white noise), this gap in quality between raw audio and transformed audio will act as a ceiling on our model's performance since we are generating spectrogram samples.

### Performance and Avenues for Improvement

As mentioned above, our performance was fairly poor on our first run. This is likely due to a few factors: we only trained for 20 epochs, we only used 1000 samples to train, and our model was overly simplistic. Luckily, these limitations also act as avenues to improve our model!

## An Improved VAE Model

