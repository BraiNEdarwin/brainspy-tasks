#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:32:25 2018
This script generates all binary assignments of N elements.
@author: hruiz and ualegre
"""
from bspytasks.benchmarks.capacity.vc_dimension_test import VCDimensionTest
from bspytasks.utils.excel import ExcelFile
from bspyalgo.utils.io import save, load_configs

import numpy as np


class CapacityTest():

    def __init__(self, configs):
        self.configs = configs
        self.current_dimension = configs['from_dimension']
        configs['algorithm_configs']['results_base_dir'] = configs['results_base_dir']
        self.excel_file = ExcelFile(configs['results_base_dir'] + 'capacity_test_results.xlsx')
        self.vcdimension_test = VCDimensionTest(configs, self.excel_file)

    def run_test(self):
        results = {}

        while True:
            print('==== VC Dimension %d ====' % self.current_dimension)
            self.vcdimension_test.init_test(self.current_dimension)
            opportunity = 0
            not_found = np.array([])
            while True:
                self.vcdimension_test.run_test(binary_labels=not_found)
                not_found = self.vcdimension_test.get_not_found_gates()
                opportunity += 1
                if (not_found.size == 0) or (opportunity >= self.configs['max_opportunities']):
                    break
            results[str(self.current_dimension)] = self.vcdimension_test.close_test()
            if not results[str(self.current_dimension)] or not self.next_vcdimension():
                return self.close_test(results)

    def close_test(self, results):
        self.excel_file.save_file()
        save(mode='configs', path=self.configs['results_base_dir'], filename='test_configs.json', data=self.configs)
        self.results = results
        return results

    def next_vcdimension(self):
        if self.current_dimension + 1 > self.configs['to_dimension']:
            return False
        else:
            self.current_dimension += 1
            return True


if __name__ == '__main__':
    capacity_test_configs = load_configs('configs/benchmark_tests/capacity_test/capacity_test_template_ga.json')

    test = CapacityTest(capacity_test_configs)
    test.run_test()