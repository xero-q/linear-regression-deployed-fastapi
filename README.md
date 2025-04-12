### Model of Linear Regression

This model was trained with data from the athletes.csv file using scikit-learn library. It can predict the weight of an athlete based on his/her height and gender.

Endpoint structure:

```
POST /predict

{
    "height": 190,
    "gender": "male"
}
```

Gender can be "male" or "female"

Output

```
{
    "weight":94.78
}
```

#### Installation

```
pip install -r requirements.txt
python run.py
```
