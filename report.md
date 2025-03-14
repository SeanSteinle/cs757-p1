# Assignment 1 Report
*Kiya Aminfar, Sean Steinle*

## The Simple VAE Model

### A Key Distinction: Raw Audio vs. Spectrograms

We started off by creating a simple VAE with only a few layers for the encoder and decoder. We also explored the audio modality since neither of us had much experience. From working with the data, we found that there is a key distinction to be made in whether audio is represented as raw audio in a .wav file or whether the audio has been transformed into a spectrogram. You can see this difference qualitatively in our `simple_vae.ipynb` notebook where we play the raw audio, a version of the raw audio which was translated into a spectrogram and back, and our predicted spectrogram translated into audio. This distinction is key, because while our first samples were very poor (essentially white noise), this gap in quality between raw audio and transformed audio will act as a ceiling on our model's performance since we are generating spectrogram samples.

### Performance and Avenues for Improvement

As mentioned above, our performance was fairly poor on our first run. This is likely due to a few factors: we only trained for 20 epochs, we only used 1000 samples to train, and our model was overly simplistic. Luckily, these limitations also act as avenues to improve our model!

### An Aside: Saving Models

Something I've accidentally spent a lot of time on is trying to save out our VAE model. This would be convenient so that we do not need to retrain our model from scratch each time we create it. However, saving Keras objects is non-trivial, and must be done carefully. Something I've quickly learned is that it's probably best to keep your model very modular as opposed to monolithic. For example, it would be better to almost treat the encoder and decoder as separate models which can be passed in to assemble the VAE object, rather than trying to serialize the entire VAE object.

Here are some threads I've been looking at for reference:
1. [Blog on how to build a VAE in Keras](https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/)
2. [Reddit thread on saving Keras VAE's](https://www.reddit.com/r/learnmachinelearning/comments/t4dbmb/how_to_save_vae_model_made_by_keras/)

## An Improved VAE Model

