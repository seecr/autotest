
import multiprocessing

is_main_process = multiprocessing.current_process().name == "MainProcess"


