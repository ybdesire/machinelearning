from allennlp.predictors.predictor import Predictor

# download model from https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz
predictor = Predictor.from_path("bidaf-model-2017.09.15-charpad.tar.gz")
r  = predictor.predict(
  passage="The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano.",
  question="Who stars in The Matrix?"
)

print(r)

'''
{'best_span_str': 'Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano', }
'''