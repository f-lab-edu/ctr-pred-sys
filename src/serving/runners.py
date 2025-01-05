import bentoml

class ModelRunner:
    """
    Runner class for model serving
    """
    
    def __init__(self, model_name, model_version = "latest"):
        """
        Initialize the model runner.

        Args:
            model_name (str): The name of the model registered in BentoML.
            model_version (str): The version of the model to use. Defaults to 'latest'.
        """
        self.model_name = model_name
        self.model_version = model_version
        self.runner = self._load_runner()

    def _load_runner(self):
        """
        Load the model runner from BentoML.

        Returns:
            Runner: The BentoML runner for the specified model.
        """
        model_ref = bentoml.picklable_model.get(f"{self.model_name}:{self.model_version}")
        print(f"Loaded model: {self.model_name}, version: {self.model_version}")
        return model_ref.to_runner()

    def get_runner(self):
        """
        Get the loaded runner.

        Returns:
            Runner: The BentoML runner.
        """
        return self.runner
