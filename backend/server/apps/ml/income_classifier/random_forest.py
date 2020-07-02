import joblib
import pandas as pd

class RandomForestClassifier:
    def __init__(self):
        '''
        Constructor: loads preprocessing objects and Random Forest object (created with Jupyter)
        '''
        # path_to_artifacts = "../../reasearch/"
        path_to_artifacts = "c:/users/ysheinfil/scedastic/my_ml_service/research/"
        self.values_fill_missing = joblib.load(path_to_artifacts + "train_mode.joblib")
        self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")
        self.model = joblib.load(path_to_artifacts + "random_forest.joblib")

    def preprocessing(self, input_data):
        '''
        Takes JSON data, converts it to pandas.DataFrame and apply preprocessing
        '''
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])

        # fill in missing values
        input_data.fillna(self.values_fill_missing)

        # convert categoricals
        for column in ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country",]:
            categorical_convert = self.encoders[column]
            input_data[column] = categorical_convert.transform(input_data[column])
        
        return input_data

    def predict(self, input_data):
        '''
        Calls ML for computing predictions on prepared data
        '''
        return self.model.predict_proba(input_data)

    def postprocesseing(self, input_data):
        '''
        Applies post-processing on prediction values
        '''
        label = "<=50K"
        if input_data[1] > 0.5:
            label = ">50K"
        return {"probability": input_data[1], "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        '''
        Combines preprocessing, predict and postprocessing and returns JSON with the response
        '''
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]    # only one sample
            prediction = self.postprocesseing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
