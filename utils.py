import joblib                                      # library used to load .pkl files saved with joblib.dump()

# Load saved model
model = joblib.load("models/svm_model.pkl")         # load the trained LinearSVC classifier from disk

# Load TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")   # load the fitted TfidfVectorizer used during training


def predict_sentiment(comment):
    """
    Predict sentiment of a Reddit comment.
    Returns:
        prediction -> class label
        score -> confidence score
    """

    vector = vectorizer.transform([comment])         # convert the raw comment string into a TF-IDF vector;
                                                      # transform() expects a list, so comment is wrapped in [ ]

    prediction = model.predict(vector)[0]            # predict the class (-1, 0, or 1); [0] unwraps the single
                                                      # result out of the array predict() returns

    score = model.decision_function(vector)          # get the raw distance-from-boundary score for each class;
                                                      # this is NOT a 0-100% confidence, it's an unbounded
                                                      # signed number per class (e.g. [[-1.2, 0.3, 2.1]])

    return prediction, score                          # send both values back to app.py