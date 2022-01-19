from score_model import score_model

if __name__ == '__main__':
    data = 'water_potability.csv'
    features_to_prepare = ['ph', 'Sulfate', 'Trihalomethanes']
    print(score_model(data, features_to_prepare))
