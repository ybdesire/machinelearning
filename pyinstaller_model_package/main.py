from sklearn.ensemble import RandomForestClassifier
import joblib as jl
from utils import x_data,y_data
import traceback


def main():
    try:
        print( 'x_data.shape={0}, y_data.shape={1}'.format( (len(x_data),len(x_data[0])), len(y_data)) )
        model = jl.load('model_rf.jl')
        for i in range(10):
            x = x_data[i]
            y_true = y_data[i]
            y_pred = model.predict([x])[0]
            print('i={0},y_pred={1},y_true={2}'.format(i,y_pred,y_true))
    except:
        msg = traceback.format_exc()
        print(msg)
    
if __name__ == '__main__':
    main()    