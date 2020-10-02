# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:10:26 2019

@author: thele
"""
import sys
import json
from .Sampler_factory import Paper_sampler #, Subsampler
from .Investigation.Investigation_factory import Investigation_stage#, Mock_Investigation_stage
from .main_utils.utils import Timer, plot_conditional_idx_improvment
from .main_utils.model_surf_utils import show_gpr_gpc, show_dummy_device, show_dev_with_points
from .Playground.mock_device import build_mock_device_with_json#, build_fake_device_with_json
import numpy as np
from . import perform_registration as pr


def tune_with_pygor_from_file(config_file):
    with open(config_file) as f:
        configs = json.load(f)


    pygor_path = configs.get('path_to_pygor', None)
    if pygor_path is not None:
        sys.path.insert(0, pygor_path)
    import Pygor
    pygor = Pygor.Experiment(xmlip=configs.get('ip', None))

    gates = configs['gates']
    plunger_gates = configs['plunger_gates']

    chan_no = configs['chan_no']

    grouped = any(isinstance(i, list) for i in gates)

    if grouped:
        def jump(params, plungers=False):

            if plungers:
                labels = plunger_gates
            else:
                labels = gates

            for i, gate_group in enumerate(labels):
                pygor.setvals(gate_group, [params[i]] * len(gate_group))

            return params
    else:
        def jump(params, plungers=False):
            # print(params)
            if plungers:
                labels = plunger_gates
            else:
                labels = gates
            pygor.setvals(labels, params)
            return params

    def measure():
        cvl = pygor.do0d()[chan_no][0]
        return cvl

    def check():
        return pygor.getvals(plunger_gates)

    assert len(gates) == len(configs['general']['origin'])

    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump, measure, check, configs['investigation'], inv_timer)

    return tune(jump, measure, investigation_stage, configs)


def tune_with_playground_from_file(config_file):
    with open(config_file) as f:
        configs = json.load(f)

    device = build_mock_device_with_json(configs['playground'])

    if configs['playground'].get('plot', False): show_dummy_device(device, configs)

    plunger_gates = configs['plunger_gates']

    def jump(params, inv=False):
        if inv:
            return params
        else:
            return device.jump(params)

    measure = device.measure

    check = lambda: device.check(plunger_gates)

    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump, measure, check, configs['investigation'], inv_timer)

    results, sampler = tune(jump, measure, investigation_stage, configs)

    fields = ['vols_pinchoff', 'conditional_idx', 'origin']

    if configs['playground'].get('plot', False):
        show_gpr_gpc(sampler.gpr, configs, *sampler.t.get(*fields), gpc=sampler.gpc.predict_comb_prob)
        plot_conditional_idx_improvment(sampler.t['conditional_idx'], configs)

    return results, sampler


def redo_with_pygor_from_file(config_file, pointcloud):
    with open(config_file) as f:
        configs = json.load(f)

    pygor_path = configs.get('path_to_pygor', None)
    if pygor_path is not None:
        sys.path.insert(0, pygor_path)
    import Pygor
    pygor = Pygor.Experiment(xmlip=configs.get('ip', None))

    gates = configs['gates']
    plunger_gates = configs['plunger_gates']

    chan_no = configs['chan_no']

    grouped = any(isinstance(i, list) for i in gates)

    if grouped:
        def jump(params, plungers=False):

            if plungers:
                labels = plunger_gates
            else:
                labels = gates

            for i, gate_group in enumerate(labels):
                pygor.setvals(gate_group, [params[i]] * len(gate_group))

            return params
    else:
        def jump(params, plungers=False):
            # print(params)
            if plungers:
                labels = plunger_gates
            else:
                labels = gates
            pygor.setvals(labels, params)
            return params

    def measure():
        cvl = pygor.do0d()[chan_no][0]
        return cvl

    def check():
        return pygor.getvals(plunger_gates)

    assert len(gates) == len(configs['general']['origin'])

    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump, measure, check, configs['investigation'], inv_timer)

    return redo(jump, measure, investigation_stage, configs)


def redo_with_playground_from_file(config_file, pointcloud):
    with open(config_file) as f:
        configs = json.load(f)

    device = build_mock_device_with_json(configs['playground'])

    if configs['playground'].get('plot', False): show_dummy_device(device, configs)

    plunger_gates = configs['plunger_gates']

    def jump(params, inv=False):
        if inv:
            return params
        else:
            return device.jump(params)

    measure = device.measure

    check = lambda: device.check(plunger_gates)

    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump, measure, check, configs['investigation'], inv_timer)

    results, sampler = redo(jump, measure, investigation_stage, configs, pointcloud)

    fields = ['vols_pinchoff', 'conditional_idx', 'origin']

    if configs['playground'].get('plot', False):
        show_gpr_gpc(sampler.gpr, configs, *sampler.t.get(*fields), gpc=sampler.gpc.predict_comb_prob)
        plot_conditional_idx_improvment(sampler.t['conditional_idx'], configs)

    return results, sampler


def tune_from_file(jump, measure, check, config_file):
    with open(config_file) as f:
        configs = json.load(f)

    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump, measure, check, configs['investigation'], inv_timer)
    results, sampler = tune(jump, measure, investigation_stage, configs)
    return results, sampler


def redo_from_file(jump, measure, check, config_file, pointcloud):
    with open(config_file) as f:
        configs = json.load(f)

    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump, measure, check, configs['investigation'], inv_timer)
    results, sampler = redo(jump, measure, investigation_stage, configs, pointcloud)
    return results, sampler


def tune(jump, measure, investigation_stage, configs):
    configs['jump'] = jump
    configs['measure'] = measure
    configs['investigation_stage_class'] = investigation_stage
    ps = Paper_sampler(configs)
    for i in range(configs['general']['num_samples']):
        print("============### ITERATION {} of {} ###============".format(i+1, configs['general']['num_samples']))
        results = ps.do_iter()
        for key, item in results.items():
            print("%s:" % (key), item[-1])

    return results, ps


def tune_origin_variable(jump, measure, par_invstage, child_invstage, par_configs, child_configs):
    par_configs['jump'], child_configs['jump'] = jump, jump
    par_configs['measure'], child_configs['measure'] = measure, measure
    par_configs['investigation_stage_class'], child_invstage['investigation_stage_class'] = par_invstage, child_invstage
    par_ps = Paper_sampler(par_configs)
    child_ps_list = []

    ps = par_ps
    par_flag = True
    for i in range(par_configs['general']['num_samples']):
        print("============### ITERATION %i ###============" % i)
        results = ps.do_iter()
        for key, item in results.items():
            print("%s:" % (key), item[-1])
        if par_flag:
            if new_origin_condition(ps):
                child_configs_new = config_constructor(child_configs, results)
                child_ps_list += [Paper_sampler(child_configs_new)]

        ps, par_flag = task_selector(par_ps, child_ps_list, i)


def redo(jump, measure, investigation_stage, configs, pointcloud):
    configs['jump'] = jump
    configs['measure'] = measure
    configs['investigation_stage_class'] = investigation_stage
    ps = Redo_sampler(configs, pointcloud)
    for i in range(configs['general']['num_samples']):
        print("============### ITERATION %i ###============" % i)
        results = ps.do_iter()
        for key, item in results.items():
            print("%s:" % (key), item[-1])

    return results, ps


def sub_tune(jump, measure, investigation_stage, configs, subsample_config, origin=None):
    # subsample_config is the key item in the configs for this subsampler. E.g. 'subsample'
    print(subsample_config)
    configs['jump'] = jump
    configs['measure'] = measure
    configs['investigation_stage_class'] = investigation_stage

    # To avoid confusion with names
    # if 'general' in configs:
    #    configs.pop('general')

    ps = Subsampler(configs, subsample_config, origin)
    for i in range(configs[subsample_config]['num_sub_samples']):
        print("============### ITERATION {} of {} ###============".format(i + 1,
                                                                          configs[subsample_config]['num_sub_samples']))
        results = ps.do_iter()
        for key, item in results.items():
            print("%s:" % (key), item[-1])

    return results, ps


# JDH addition - can be removed after testing
# TODO: remove after testing
def show_device_shape(config_file):
    with open(config_file) as f:
        configs = json.load(f)

    device = build_mock_device_with_json(configs['playground'])

    # Eventually revert this to normal show_dummy_device, remove import at top of page
    show_dev_with_points(device, configs)

    return device



# Original way of doing this. It's not very good.
def subtune_routine(config_file, config_subset='subsample'):
    with open(config_file) as f:
        configs = json.load(f)

    device = build_fake_device_with_json(configs['playground'])
    sub_slice = 2

    def sub_jump(params_sub):
        params_full = np.asarray(device.params).squeeze()

        params_full[:sub_slice] = params_sub
        device.jump(params_full.squeeze())

        return params_full

    measure = device.measure
    dots = device.fake_dots

    plunger_gates = configs['plunger_gates']

    check = lambda: device.check(plunger_gates)

    def subroutine(config_subset, origin=None):
        inv_timer = Timer()
        investigation_stage = Mock_Investigation_stage(sub_jump, measure, check, configs, inv_timer, dots,
                                                       start_params=device.params)
        results, sampler = sub_tune(sub_jump, measure, investigation_stage, configs, config_subset, origin)
        fields = ['vols_pinchoff', 'conditional_idx', 'origin']
        p_offs = results['vols_pinchoff']
        points = np.concatenate(p_offs).reshape(-1, 2)
        return points

    def move_and_measure(params, plot=False):
        device.set_params(params)
        print(device.params)
        print("Moved to ", device.params)

        points = np.array(subroutine(config_subset)).squeeze()

        return points

    ####################################################

    # Voltage values at which gates will be measured
    x = [0, -100 , -200, -450, -600]
    #x = [0, -200, -400]

    # Indices of gates to be measured
    gates = [2, 3, 4]

    full_points = []
    transforms = []

    for gate in gates:
        out = []
        voltages = [0] * device.n
        for vol in x:
            voltages[gate] = vol
            print(voltages)
            points = move_and_measure(voltages)
            out.append(points)
        full_points.append(out)

    for gate_set in full_points:
        out = []
        for point_set in gate_set:
            _transform = np.array(pr.scaling_registration(np.array(gate_set[0]).T, np.array(point_set).T))
            out.append(_transform)
        transforms.append(out)

    val_x = [0, 0, -330, -39, -473]

    val_points = move_and_measure(val_x)

    validator = np.array(pr.scaling_registration(np.array(gate_set[0]).T, np.array(val_points).T))

    return full_points, transforms, x, validator, val_points


if __name__ == '__main__':
    pass
    # tune_with_pygor_from_file('tuning_config.json')
