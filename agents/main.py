import cnn_agent as cnn
import cnn2 as cnn2
import cnn3 as cnn3
import cnn4 as cnn4
import Testing as testing
import random_agents as ra


numberOfGenerations = 5
currentGeneration = 0
sample_count = 200
numberOfUpdateSamples = 100
numberOfEpochs = 200
batch_size = 8
#ra.deleteSamples()
#ra.deleteTestSamples()
#ra.deleteUpdatedSamples()
ra.storeTrainSamples(sample_count)
cnn.train_model(numberOfEpochs,batch_size, is_update=False)
#cnn2.train_model(numberOfEpochs,batch_size,is_update=False)
#cnn3.train_model(numberOfEpochs,batch_size,is_update=False)
#cnn4.train_model(numberOfEpochs,batch_size,is_update=False)
def testing():
    for i in range(numberOfGenerations):
        currentGeneration = currentGeneration + 1
        for j in range(numberOfUpdateSamples):
            ra.deleteTestSamples()
            ra.storeTestSamples(sample_count)
            testing.add_samples_to_trained_model(currentGeneration)

        cnn.train_model(numberOfEpochs, is_update=True)


