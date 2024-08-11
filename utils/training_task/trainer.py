import os
from .trainer_yolov8_detection import start_training_yolo_detection_task, read_yolo_loss_values
from .trainer_detectron2_instance_segmentation import start_training_detectron2_task, read_detectron2_metrics

# Define constants for algorithm and framework names
OBJECT_DETECTION = 'ObjectDetection'
INSTANCE_SEGMENTATION = 'InstanceSegmentation'
YOLOV8 = 'YOLOv8'
DETECTRON2_INSTANCE_SEGMENTATION = 'Detectron2-InstanceSegmentation'

def get_trainer(algo_name: str, framework_name: str):
    """
    Returns the corresponding trainer function based on the algorithm and framework names.
    
    Parameters:
    algo_name (str): The name of the algorithm, e.g., 'ObjectDetection'.
    framework_name (str): The name of the framework, e.g., 'YOLOv8'.
    
    Returns:
    function: The function that starts the training task for the specified algorithm and framework.
    
    Raises:
    NotImplementedError: If the combination of algorithm and framework is not implemented.
    """
    if algo_name == OBJECT_DETECTION and framework_name == YOLOV8:
        return start_training_yolo_detection_task
    elif algo_name == INSTANCE_SEGMENTATION and framework_name == DETECTRON2_INSTANCE_SEGMENTATION:
        return start_training_detectron2_task
    else:
        raise NotImplementedError(f"Trainer not implemented for algorithm: {algo_name}, framework: {framework_name}")

def get_loss_parser(algo_name: str, framework_name: str):
    """
    Returns the corresponding loss parser function based on the algorithm and framework names.
    
    Parameters:
    algo_name (str): The name of the algorithm, e.g., 'ObjectDetection'.
    framework_name (str): The name of the framework, e.g., 'YOLOv8'.
    
    Returns:
    function: The function that reads the loss values or metrics for the specified algorithm and framework.
    
    Raises:
    NotImplementedError: If the combination of algorithm and framework is not implemented.
    """
    if algo_name == OBJECT_DETECTION and framework_name == YOLOV8:
        return read_yolo_loss_values
    elif algo_name == INSTANCE_SEGMENTATION and framework_name == DETECTRON2_INSTANCE_SEGMENTATION:
        return read_detectron2_metrics
    else:
        raise NotImplementedError(f"Loss parser not implemented for algorithm: {algo_name}, framework: {framework_name}")

def get_loss_file(algo_name: str, framework_name: str, training_project_root: str, user_id: str, save_key: str) -> str:
    """
    Constructs and returns the path to the loss file based on the algorithm and framework names.
    
    Parameters:
    algo_name (str): The name of the algorithm, e.g., 'ObjectDetection'.
    framework_name (str): The name of the framework, e.g., 'YOLOv8'.
    training_project_root (str): The root directory of the training project.
    user_id (str): The ID of the user.
    save_key (str): A unique key for saving the training results.
    
    Returns:
    str: The full path to the loss file.
    
    Raises:
    NotImplementedError: If the combination of algorithm and framework is not implemented.
    """
    if algo_name == OBJECT_DETECTION and framework_name == YOLOV8:
        return os.path.join(training_project_root, user_id, save_key, 'project', 'exp', 'results.csv')
    elif algo_name == INSTANCE_SEGMENTATION and framework_name == DETECTRON2_INSTANCE_SEGMENTATION:
        return os.path.join(training_project_root, user_id, save_key, 'project', 'metrics.json')
    else:
        raise NotImplementedError(f"Loss file path not implemented for algorithm: {algo_name}, framework: {framework_name}")
