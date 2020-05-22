from allennlp.predictors.predictor import Predictor


# take a pair of sentences and predict whether the facts in the first necessarily imply the facts in the second one

# Firstly, download model from https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz
predictor = Predictor.from_path("decomposable-attention-elmo-2018.02.19.tar.gz")
result = predictor.predict(   
  hypothesis="Two women are drinking.",
  premise="Two women are wandering along the shore drinking iced tea."
)


print(result)


'''
'label_probs': [0.9865285158157349, 0.0020866768900305033, 0.011384865269064903]

Entailment : 98.6%
Contradiction : 0.2%
Neutral : 1.13%
'''
