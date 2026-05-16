import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob
import math

def bandpass_filter(data, fs, low=0.1, high=45):
    b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return signal.filtfilt(b, a, data)

def notch_filter(data, fs, freq=4):
    b, a = signal.iirnotch(freq/(fs/2), Q=10)
    return signal.filtfilt(b, a, data)

def bandstop_filter(data, fs, low=3, high=5, order=4):
    b, a = signal.butter(
        order,
        [low/(fs/2), high/(fs/2)],
        btype='bandstop'
    )
    return signal.filtfilt(b, a, data)


def plot_all_students_single_task(dataset_path, student_ids, task_id, round_id, start_sec, end_sec, fs=512):
    
    task_names = {1: "Relax", 2: "Focus", 3: "Blink"}
    task_name = task_names.get(task_id, f"Task {task_id}")

    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"EEG Segment Comparison: {task_name} (Round {round_id})", fontsize=16)

    for student_id in student_ids:
        pattern = os.path.join(dataset_path, student_id, f"*_{task_id}_{round_id}.txt")
        files = glob.glob(pattern)
        
        if files:
            try:
                raw_data = np.loadtxt(files[0])
                # =============================================
                # ==== adding filters  ====
                # raw_data = bandpass_filter(raw_data, fs)
                # raw_data = notch_filter(raw_data, fs)
                # raw_data = bandstop_filter(raw_data, fs)
                # =============================================
                actual_end = min(end_idx, len(raw_data))
                data_slice = raw_data[start_idx:actual_end]
                time = np.arange(start_idx, actual_end) / fs
                
                ax.plot(time, data_slice, lw=1.0, alpha=0.8, label=student_id)
                
            except Exception as e:
                print(f"Error loading {student_id}: {e}")
        else:
            print(f"File not found for: {student_id}")

    ax.set_title(f"Time: {start_sec}s to {end_sec}s", fontsize=12)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.set_ylim(-1000, 1000) 
    ax.grid(True, alpha=0.3)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()

def plot_students_separate_subplots(dataset_path, student_ids, task_id, round_id, start_sec, end_sec, fs=512):
    
    task_names = {1: "Relax", 2: "Focus", 3: "Blink"}
    task_name = task_names.get(task_id, f"Task {task_id}")

    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    cols = 3
    rows = math.ceil(len(student_ids) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows), sharex=True, sharey=True)
    fig.suptitle(f"EEG Segments: {task_name} (Round {round_id}) | {start_sec}s - {end_sec}s", fontsize=16)

    axes_flat = axes.flatten()

    for i, student_id in enumerate(student_ids):
        ax = axes_flat[i]
        pattern = os.path.join(dataset_path, student_id, f"*_{task_id}_{round_id}.txt")
        files = glob.glob(pattern)
        
        if files:
            try:
                raw_data = np.loadtxt(files[0])
                # =============================================
                # ==== adding filters =========================
                # raw_data = bandpass_filter(raw_data, fs)
                # raw_data = notch_filter(raw_data, fs)
                # raw_data = bandstop_filter(raw_data, fs)
                # =============================================
                actual_end = min(end_idx, len(raw_data))
                data_slice = raw_data[start_idx:actual_end]
                time = np.arange(start_idx, actual_end) / fs
                
                ax.plot(time, data_slice, lw=0.8, color='#1f77b4')
                ax.set_title(student_id, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, "Load Error", ha='center', va='center', fontsize=10, color='red')
        else:
            ax.text(0.5, 0.5, "File Not Found", ha='center', va='center', fontsize=10, color='gray')

    for ax in axes_flat:
        ax.set_ylim(-1000, 1000)

    for j in range(len(student_ids), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_single_subject_all_rounds(dataset_path, student_id, task_id, rounds, start_sec, end_sec, fs=512):
    
    task_names = {1: "Relax", 2: "Focus", 3: "Blink"}
    task_name = task_names.get(task_id, f"Task {task_id}")

    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    cols = 3
    rows = math.ceil(len(rounds) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows), sharex=True, sharey=True)
    fig.suptitle(f"EEG Progression: {student_id} | {task_name} (All Rounds) | {start_sec}s - {end_sec}s", fontsize=16)

    axes_flat = axes.flatten()

    for i, round_id in enumerate(rounds):
        ax = axes_flat[i]
        pattern = os.path.join(dataset_path, student_id, f"*_{task_id}_{round_id}.txt")
        files = glob.glob(pattern)
        
        if files:
            try:
                raw_data = np.loadtxt(files[0])
                # =============================================
                # ==== adding filters =========================
                # raw_data = bandpass_filter(raw_data, fs)
                # raw_data = notch_filter(raw_data, fs)
                # raw_data = bandstop_filter(raw_data, fs)
                # =============================================
                actual_end = min(end_idx, len(raw_data))
                data_slice = raw_data[start_idx:actual_end]
                time = np.arange(start_idx, actual_end) / fs
                
                ax.plot(time, data_slice, lw=0.8, color='#2ca02c')
                ax.set_title(f"Round {round_id}", fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, "Load Error", ha='center', va='center', fontsize=10, color='red')
        else:
            ax.text(0.5, 0.5, "File Not Found", ha='center', va='center', fontsize=10, color='gray')

    for ax in axes_flat:
        ax.set_ylim(-1000, 1000)

    for j in range(len(rounds), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_task_fixed_y(dataset_path, student_id, rounds, start_sec, end_sec, fs=512):

    tasks = [1, 2, 3]
    # rounds = [17, 18, 19]
    task_names = {1: "Relax", 2: "Focus", 3: "Blink"}

    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    
    fig, axes = plt.subplots(len(rounds), 3, figsize=(12, 2 * len(rounds)), sharex=True, sharey=True)
    fig.suptitle(f"EEG Segment (Fixed Y): {student_id}", fontsize=14)

    
    for j, round_id in enumerate(rounds):
        for i, task_id in enumerate(tasks):
            ax = axes[j, i]
            pattern = os.path.join(dataset_path, student_id, f"*_{task_id}_{round_id}.txt")
            files = glob.glob(pattern)
            
            if files:
                try:
                    raw_data = np.loadtxt(files[0])
                    # =============================================
                    # ==== adding filters =========================
                    # raw_data = bandpass_filter(raw_data, fs)
                    # raw_data = notch_filter(raw_data, fs)
                    # raw_data = bandstop_filter(raw_data, fs)
                    # =============================================
                    actual_end = min(end_idx, len(raw_data))
                    data_slice = raw_data[start_idx:actual_end]
                    time = np.arange(start_idx, actual_end) / fs
                    
                    ax.plot(time, data_slice, lw=0.6, color='C'+str(i))
                    ax.set_title(f"{task_names[task_id]} R{round_id}", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    
                    ax.set_ylim(-1000, 1000) 
                    
                except Exception as e:
                    ax.text(0.5, 0.5, "Load Error", ha='center', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



if __name__ == "__main__":

    student_ids = [f"S{str(i).zfill(2)}" for i in range(1, 19)]
    student_ids.append("b12901035")
    student_ids.append("b12901028")
    dataset_path = 'bci_dataset_114-2_any'
    target_task = 3    # 1: Relax, 2: Focus, 3: Blink
    target_round = 1  
    
    start_sec = 10
    end_sec = 20

    ###############################################################
    # 1. All students, single task, single round
    ###############################################################
    plot_students_separate_subplots(dataset_path, student_ids, target_task, target_round, start_sec, end_sec)
    
    ###############################################################
    # 2. Single student, single task, all rounds
    ###############################################################
    # plot_single_subject_all_rounds(dataset_path, "S04", target_task, rounds=range(1, 19), start_sec=start_sec, end_sec=end_sec)

    ###############################################################
    # 3. Single student, all tasks, all rounds 
    ###############################################################
    #plot_task_fixed_y(dataset_path, "S01", rounds=[11, 12, 13 , 14, 15 ,16], start_sec=0, end_sec=20)