# Language Modelling by Smoothing and LSTMs

## Language Modelling using Smoothing:

There are two types of smoothing techniques used in language modelling:
- Kneser-Ney Smoothing
- Witten-Bell Smoothing

To run the application, run the tokeniser file, with the command:

```
python3 tokeniser.py
```

Depending on the code, the function will run accordingly. Initially it is set to Kneser-Ney smoothing. To change it to Witten-Bell smoothing, change what the probability function is calling.

## Language Modelling using LSTMs:

To run the application, run the LSTM file, with the command:

```
python3 lm.py
```

The code will run the LSTM model, and will output the probability of the model. The LSTM models are pre-trained, and the code will load the model from the respective files. To change the model, simply change the file name in the code.

## Perplexity:

All the perplexity scores of all the models are in the respective files. A key that we can use:
- 1 = Kneser-Ney, Pride and Prejudice
- 2 = Kneser-Ney, Ulysses
- 3 = Witten-Bell, Pride and Prejudice
- 4 = Witten-Bell, Ulysses
- 5 = LSTM, Pride and Prejudice
- 6 = LSTM, Ulysses