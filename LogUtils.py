import csv
import os
import platform
import sys

import torch


def get_script_name():
    """ Pobiera nazwę skryptu (bez rozszerzenia) """
    return os.path.splitext(os.path.basename(sys.argv[0]))[0]


def log_to_file(log_data, file_path):
    """ Zapisuje logi do pliku CSV """
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["Time", "Epoch", "Train Loss", "Val Loss", "Accuracy"])
        writer.writerow(log_data)

def get_hardware_info():
    """ Pobiera informacje o sprzęcie: CPU, GPU i system operacyjny """
    info = {
        "System": platform.system(),
        "Wersja systemu": platform.version(),
        "Procesor": platform.processor(),
        "Architektura": platform.architecture()[0],
        "Ilość rdzeni CPU": os.cpu_count(),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Brak GPU"
    }
    return info

def log_hardware_info(file_path):
    """ Zapisuje informacje o sprzęcie do pliku """
    with open(file_path, mode='a', newline='') as file:
        file.write("=== HARDWARE AND SOFTWARE INFO ===\n")
        for key, value in get_hardware_info().items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
