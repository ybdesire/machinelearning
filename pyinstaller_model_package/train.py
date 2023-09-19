from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import joblib as jl


def main():
    data = load_breast_cancer()
    x_data = data.data
    y_data = data.target
    print( 'x_data.shape={0}, y_data.shape={1}'.format(x_data.shape, y_data.shape) )
    model = RandomForestClassifier()
    model.fit(x_data,y_data)
    jl.dump(model, 'model_rf.jl')
    print(x_data[:10,:])
    
if __name__ == '__main__':
    main()    