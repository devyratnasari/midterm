import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger

app=Flask(__name__)

Swagger(app)

model=pickle.load(open("logistic_regression.pkl","rb"))

@app.route('/predict', methods=["GET"])
def predict_class():
    
    """predicting breast cancer detection using logistic regression by parameter
    ---
    parameters:  
      - name: radius_mean
        in: query
        type: number
        required: true
      - name: texture_mean
        in: query
        type: number
        required: true
      - name: perimeter_mean
        in: query
        type: number
        required: true
      - name: area_mean
        in: query
        type: number
        required: true
      - name: smoothness_mean
        in: query
        type: number
        required: true
      - name: compactness_mean
        in: query
        type: number
        required: true
      - name: concavity_mean
        in: query
        type: number
        required: true
      - name: concave points_mean
        in: query
        type: number
        required: true
        - name: symmetry_mean
        in: query
        type: number
        required: true
      - name: fractal_dimension_mean
        in: query
        type: number
        required: true
      - name: texture_se
        in: query
        type: number
        required: true
      - name: perimeter_se
        in: query
        type: number
        required: true
      - name: area_se
        in: query
        type: number
        required: true
      - name: smoothness_se
        in: query
        type: number
        required: true
      - name: compactness_se
        in: query
        type: number
        required: true
      - name: concavity_se
        in: query
        type: number
        required: true
      - name: concave points_se
        in: query
        type: number
        required: true
      - name: symmetry_se
        in: query
        type: number
        required: true
      - name: fractal_dimension_se
        in: query
        type: number
        required: true
      - name: radius_worst
        in: query
        type: number
        required: true
      - name: texture_worst
        in: query
        type: number
        required: true
      - name: perimeter_worst
        in: query
        type: number
        required: true
      - name: area_worst
        in: query
        type: number
        required: true
      - name: smoothness_worst
        in: query
        type: number
        required: true
      - name: compactness_worst
        in: query
        type: number
        required: true
      - name: concavity_worst
        in: query
        type: number
        required: true
      - name: concave points_worst
        in: query
        type: number
        required: true
      - name: symmetry_worst
        in: query
        type: number
        required: true
      - name: fractal_dimension_worst
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    radius_mean =request.args.get('radius_mean')
    texture_mean=request.args.get('texture_mean')
    perimeter_mean=request.args.get('perimeter_mean')
    area_mean=request.args.get('area_mean')
    smoothness_mean =request.args.get('smoothness_mean')
    compactness_mean=request.args.get('compactness_mean')
    concavity_mean=request.args.get('concavity_mean')
    concave_points_mean=request.args.get('concave_points_mean')  
    symmetry_mean =request.args.get('symmetry_mean')
    fractal_dimension_mean=request.args.get('fractal_dimension_mean')
    radius_se=request.args.get('radius_se')
    texture_se=request.args.get('texture_se')
    perimeter_se =request.args.get('perimeter_se')
    area_se=request.args.get('area_se')
    smoothness_se=request.args.get('smoothness_se')
    compactness_se=request.args.get('compactness_se')    
    concavity_se =request.args.get('concavity_se')
    concave_points_se=request.args.get('concave_points_se')
    symmetry_se=request.args.get('symmetry_se')
    fractal_dimension_se =request.args.get('smoothness_fractal_dimension_semean')
    radius_worst=request.args.get('radius_worst')
    texture_worst=request.args.get('texture_worst')
    perimeter_worst=request.args.get('perimeter_worst')
    area_worst=request.args.get('area_worst')  
    smoothness_worst =request.args.get('smoothness_worst')
    compactness_worst=request.args.get('compactness_worst')
    concavity_worst=request.args.get('concavity_worst')
    concave_points_worst=request.args.get('concave_points_worst')
    symmetry_worst =request.args.get('symmetry_worst')
    fractal_dimension_worst=request.args.get('fractal_dimension_worst')
    prediction=model.predict([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])
    return " The Predicated Class is"+ str(prediction)

@app.route('/predict_test', methods=["POST"])
def predict_test_class():
    
    """predicting breast cancer detection using csv file
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    test_csv=pd.read_csv(request.files.get("file"))
    prediction=model.predict(test_csv)
    return " The Predicated Class for the Test csv is"+ str(list(prediction))


if __name__=='__main__':
    app.run()