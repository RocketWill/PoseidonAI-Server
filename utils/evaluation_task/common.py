from enum import Enum, auto

class EvaluationError(Enum):
    ASSERTION_FAILED = auto()
    FILE_NOT_FOUND = auto()
    MEMORY_ERROR = auto()
    RUNTIME_ERROR = auto()
    CUDA_OUT_OF_MEMORY = auto()
    CUDNN_RNN_BACKWARD_ERROR = auto()
    UNEXPECTED_ERROR = auto()

class EvaluationStatus(Enum):
    IDLE = auto
    PENDING = auto()
    PROCESSING = auto()
    SUCCESS = auto()
    FAILURE = auto()
    ERROR = auto()
    REVOKED = auto()

