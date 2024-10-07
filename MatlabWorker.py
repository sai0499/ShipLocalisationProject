import matlab.engine
from PyQt5.QtCore import QThread, pyqtSignal

# Worker class to run MATLAB script in the background
class MATLABWorker(QThread):
    finished = pyqtSignal()  # Signal to emit when MATLAB processing is done
    error = pyqtSignal(str)  # Signal to emit if an error occurs

    def __init__(self, matlab_function):
        super().__init__()
        self.matlab_function = matlab_function  # This is the MATLAB function to run

    def run(self):
        try:
            # Start MATLAB and run the function
            eng = matlab.engine.start_matlab()
            if self.matlab_function == 'particle_filter':
                eng.ShipPositioningPF(nargout=0)
            elif self.matlab_function == 'ukf':
                eng.ShipPositioningUKF(nargout=0)
            self.finished.emit()  # Signal that MATLAB has finished
        except Exception as e:
            self.error.emit(str(e))  # Signal the error if something went wrong
