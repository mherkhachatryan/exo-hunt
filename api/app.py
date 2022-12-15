from hunter.preprocessing import ExoHuntDataset
from hunter.model import ExoHuntModel
from hunter.main import configs

from flask import Flask, request, jsonify, render_template, make_response

application = Flask("exo_hunter")

if configs.run_mode == "test":
    model = ExoHuntModel(model_path=configs.saved_model_path)
    dataset = ExoHuntDataset(configs.data_path, configs.random_state)
    X_test, y_test = dataset.load_test_data()
elif configs.run_mode == "train":

    raise RuntimeError("It's currently impossible to train model via API.")

template_path = "templates/index.html"


@application.route('/')
def home():
    return render_template(template_path)


@application.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    X_test, _ = dataset.load_test_data()
    classification_result = model.predict(X_test[0].reshape(1, -1))  # TODO make random

    if classification_result == 0:
        return render_template(template_path, prediction_text=f'Star is non-exoplanet system')
    else:
        return render_template(template_path, prediction_text=f'Star is exoplanet system. You found a planet!')


@application.route('/predict_api', methods=['POST'])
def predict_api():
    """
    For direct API calls trought request
    """
    request_body = request.get_json(force=True)
    planetary_system_number = request_body["planetary_system"]

    if planetary_system_number > len(X_test):
        return make_response(f"Planetary system {planetary_system_number} not found, try lowering system number",
                             404)

    prediction = model.predict(X_test[planetary_system_number].reshape(1, -1))
    prediction = "Yes" if prediction == 1 else "No"

    output = {"Is exoplanet system": prediction, "message": "Success"}

    return jsonify(output)


if __name__ == '__main__':
    application.run(debug=True)
