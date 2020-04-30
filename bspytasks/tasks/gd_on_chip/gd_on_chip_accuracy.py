import numpy as np
from bspyalgo.utils.io import create_directory_timestamp, save
from bspyalgo.utils.gd_on_chip.accuracy_tests import convergence, multiwave_accuracy, multiwave_quantitative, multiwave_accuracy_amplitudes

if __name__ == '__main__':
    """
    Runs all accuracy tests. The amount of times that a test is run can be specified in the corresponding config file.
    """
    from bspyalgo.utils.io import load_configs
    
    configs_convergence = load_configs('/configs/tasks/gd_on_chip/configs_template_on_chip_convergence.json')
    configs_multiwave = load_configs('/configs/tasks/gd_on_chip/configs_template_on_chip_multiwave_accuracy.json')
    configs_quantitative = load_configs('/configs/tasks/gd_on_chip/configs_template_on_chip_multiwave_quantitative.json')
    configs_amplitudes = load_configs('/configs/tasks/gd_on_chip/configs_template_on_chip_multiwave_accuracy_amplitudes.json')

    for mult_exps in range(configs_multiwave["nr_experiments"]):
        inputs = np.random.random(configs_multiwave["processor"]["input_electrode_no"]) * (np.array(configs_multiwave["waveform"]["input_range"])[:, 1] - np.array(configs_multiwave["waveform"]["input_range"])[:, 0]) + np.array(configs_multiwave["waveform"]["input_range"])[:, 0]
        input_waveforms, outputs, IV_gradients = multiwave_accuracy(inputs, configs_multiwave)
        save('pickle', configs_multiwave["results_path"],configs_multiwave["experiment_name"], 
                                                data = {"config":configs_multiwave,
                                                        "input_waveforms": input_waveforms,
                                                        "outputs": outputs,
                                                        "IV_gradients": IV_gradients,
                                                        "inputs": inputs
                                                        })
        del input_waveforms, outputs, IV_gradients # Delete arrays to save space for next experiments.

    for convergence_exps in range(configs_convergence["nr_experiments"]):
        inputs = np.random.random(configs_convergence["processor"]["input_electrode_no"]) * (np.array(configs_convergence["waveform"]["input_range"])[:, 1] - np.array(configs_convergence["waveform"]["input_range"])[:, 0]) + np.array(configs_convergence["waveform"]["input_range"])[:, 0]
        input_waveforms, outputs, IV_gradients, sample_times = convergence(inputs, configs_convergence)
        save('pickle', configs_convergence["results_path"],configs_convergence["experiment_name"], 
                                                data = {"config": configs_convergence,
                                                        "input_waveforms": input_waveforms,
                                                        "outputs": outputs,
                                                        "IV_gradients": IV_gradients,
                                                        "inputs": inputs,
                                                        "sample_times": sample_times
                                                        })
        del input_waveforms, outputs, IV_gradients, sample_times

    for quant_exps in range(configs_quantitative["nr_experiments"]):
        input_waveforms, outputs, IV_gradients, controls, t = multiwave_quantitative(configs_quantitative)
        save('pickle', configs_quantitative["results_path"], configs_quantitative["experiment_name"], 
                                                data = {"config": configs_quantitative,
                                                        "input_waveforms": input_waveforms,
                                                        "outputs": outputs,
                                                        "IV_gradients": IV_gradients,
                                                        "controls": controls,
                                                        "t": t
                                                        })
        del input_waveforms, outputs, IV_gradients, controls, t

    for multiwave_amps_exps in range(configs_amplitudes["nr_experiments"]):
        inputs = np.random.random(configs_amplitudes["processor"]["input_electrode_no"]) * (np.array(configs_amplitudes["waveform"]["input_range"])[:, 1] - np.array(configs_amplitudes["waveform"]["input_range"])[:, 0]) + np.array(configs_amplitudes["waveform"]["input_range"])[:, 0]
        input_waveforms, outputs, IV_gradients, t = multiwave_accuracy_amplitudes(inputs, configs_amplitudes)
        save('pickle', configs_amplitudes["results_path"], configs_amplitudes["experiment_name"], 
                                                data = {"config": configs_amplitudes,
                                                        "input_waveforms": input_waveforms,
                                                        "outputs": outputs,
                                                        "IV_gradients": IV_gradients,
                                                        "controls": inputs,
                                                        "t": t
                                                        })
        del input_waveforms, outputs, IV_gradients, t