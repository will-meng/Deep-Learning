# Training
BATCH_SIZE = 33
BATCH_STRING_LENGTH = 32
FNAME = "21839.txt"
#GRADIENT_CLIP = 1e-7
LEARNING_RATE = 0.001
NUM_CHARS = 128
NUM_EPOCHS = 100
LAYER1_SIZE = 128
LAYER2_SIZE = 129

# Generation
BURN_IN_CHARS = 2 ** 13
CHARS_TO_GENERATE = 2 ** 11

# Got ~30% accuracy with no hidden layer.
# Got ~50% accuracy with two layers of 32/128/129
# Not clear that 32/512/513 is better...
