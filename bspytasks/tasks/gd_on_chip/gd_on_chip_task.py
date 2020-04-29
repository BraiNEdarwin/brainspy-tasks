import numpy as np
import matplotlib.pyplot as plt
from bspyalgo.utils.io import create_directory_timestamp, save
from bspyalgo.algorithms.gradient.gd_on_chip import OnChipGD
from bspyalgo.algorithms.gradient.core import input_tasks     

def task_selector(task, configs):
    if configs["task_type"] == "Boolean":
        return input_tasks.booleanLogic(task, 
            configs['waveform']['input_cases'] * configs['waveform']['amplitude_lengths'],
            configs['waveform']['slope_lengths'] , configs['processor']['sampling_frequency'])
    elif configs["task_type"] == "featureExtractor":
        return input_tasks.featureExtractor(task,
            configs['waveform']['input_cases'] * configs['waveform']['amplitude_lengths'],
            configs['waveform']['slope_lengths'] , configs['processor']['sampling_frequency'])
    else: print("Error: task not regonized.")

if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    configs = load_configs('/configs/tasks/gd_on_chip/configs_template_gd_on_chip.json')
    task = configs['task']
    best_error = 1E6*np.ones(len(task))
    
    for runs in range(configs['hyperparameters']['initializations']):
        for experiments in range(len(task)):
            # If the task has not been solved yet in previous attempts, try again.
            if best_error[experiments] > configs['hyperparameters']['stop_threshold']:
                configs['task'] = task[experiments]
                GD = OnChipGD(configs)
                print('Optimizing ' + str(task[experiments]))
                # Generate input data for the defined task: 
                t, inputs, mask, targets = task_selector(task[experiments], configs)

                data = GD.optimize(inputs, targets, mask=mask)
                
                if data.results['best_error'] < best_error[experiments]:
                    best_error[experiments] = data.results['best_error']

                if configs["save_figures"]:
                    # save figure of error and of last generation
                    fig = plt.figure()
                    plt.plot(data.results['error'])
                    fig.savefig(data.path + '/error.png')
                    plt.close()

                    time = np.linspace(0, len(data.results['best_outputs'][data.results['ramp_mask']][data.results['mask']])/configs['processor']['sampling_frequency'] , len(data.results['best_outputs'][data.results['ramp_mask']][data.results['mask']]))
                    fig = plt.figure()
                    plt.plot(time, data.results['best_outputs'][data.results['ramp_mask']][data.results['mask']])
                    fig.savefig(data.path + '/best_output.png')
                    plt.close()
            else: print("Task " + str(task[experiments]) + " already solved, continuing to next task...")

    print("best errors: " + str(best_error))