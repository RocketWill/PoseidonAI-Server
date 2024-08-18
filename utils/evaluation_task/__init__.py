from utils.evaluation_task.evaluation_yolov8_detection import start_eval_yolov8_detection_task
from utils.evaluation_task.evaluation_detectron2_segmentation import start_eval_detectron2_instance_segmentation_task


# Define constants for algorithm and framework names
OBJECT_DETECTION = 'ObjectDetection'
INSTANCE_SEGMENTATION = 'InstanceSegmentation'
YOLOV8 = 'YOLOv8'
DETECTRON2_INSTANCE_SEGMENTATION = 'Detectron2-InstanceSegmentation'

def get_evaluator(algo_name: str, framework_name: str):
    """
    Returns the corresponding trainer function based on the algorithm and framework names.
    
    Parameters:
    algo_name (str): The name of the algorithm, e.g., 'ObjectDetection'.
    framework_name (str): The name of the framework, e.g., 'YOLOv8'.
    
    Returns:
    function: The function that starts the task evaluation for the specified algorithm and framework.
    
    Raises:
    NotImplementedError: If the combination of algorithm and framework is not implemented.
    """
    if algo_name == OBJECT_DETECTION and framework_name == YOLOV8:
        return start_eval_yolov8_detection_task
    elif algo_name == INSTANCE_SEGMENTATION and framework_name == DETECTRON2_INSTANCE_SEGMENTATION:
        return start_eval_detectron2_instance_segmentation_task
    else:
        raise NotImplementedError(f"Evaluator not implemented for algorithm: {algo_name}, framework: {framework_name}")