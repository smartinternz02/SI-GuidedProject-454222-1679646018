from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('./concrete_data.csv', low_memory=False)

# Split data into features and target
X = data[['cement', 'blast furnace slag', 'fly ash', 'water', 'superplasticizer', 'coarse aggregate',
          'fine aggregate',
          'age']]
y = data['strength']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

app = Flask(__name__, template_folder='templates')


@app.route("/", methods=["POST","GET"])
def predict():
    if request.method == 'POST':
        # Load the CSV file from a local path into a pandas dataframe

        cement = float(request.form["cement"])
        blast_furnace_slag = float(request.form["blast_furnace_slag"])
        fly_ash = float(request.form["fly_ash"])
        water = float(request.form["water"])
        super_plasticizer = float(request.form["super_plasticizer"])
        coarse_aggregate = float(request.form["coarse_aggregate"])
        fine_aggregate = float(request.form["fine_aggregate"])
        age = float(request.form["age"])

        # Make prediction
        new_data = pd.DataFrame({
            'cement': [cement],
            'blast furnace slag': [blast_furnace_slag],
            'fly ash': [fly_ash],
            'water': [water],
            'superplasticizer': [super_plasticizer],
            'coarse aggregate': [coarse_aggregate],
            'fine aggregate': [fine_aggregate],
            'age': [age]
        })

        prediction = model.predict(new_data)
        return render_template("prediction.html", prediction=prediction[0])
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run()
