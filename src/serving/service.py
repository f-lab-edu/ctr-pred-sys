import bentoml

from bentoml.io import NumpyNdarray
from .runners import ModelRunner

class ModelService:
    """
    Servce class for model serving
    """
    
    def __init__(self, service_name, model_name):
        """
        Initialize the BentoML service.

        Args:
            service_name (str): The name of the BentoML service.
            model_name (str): The name of the model registered in BentoML.
        """
        self.service_name = service_name
        self.model_name = model_name
        self.runner = self._create_runner()
        self.service = self._create_service()

    def _create_runner(self):
        """
        Create the model runner using the ModelRunner class.

        Returns:
            Runner: The BentoML runner.
        """
        runner_instance = ModelRunner(self.model_name)
        return runner_instance.get_runner()

    def _create_service(self):
        """
        Create the BentoML service with the associated runner.

        Returns:
            Service: The BentoML service.
        """
        svc = bentoml.Service(self.service_name, runners=[self.runner])

        @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
        def predict(input_data):
            """
            API endpoint for making predictions.

            Args:
                input_data (numpy.ndarray): Input data for prediction.

            Returns:
                numpy.ndarray: Prediction results.
            """
            return self.runner.predict.run(input_data)

        return svc

    def get_service(self):
        """
        Get the BentoML service instance.

        Returns:
            Service: The BentoML service instance.
        """
        return self.service
