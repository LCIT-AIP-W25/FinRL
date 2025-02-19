
def save_model(model, model_name="a2c_trained_model"):
    """
    Save the trained model to disk.
    """
    model.save(f"model/{model_name}.zip")
    return f"Model saved as {model_name}.zip"
